"""Bounded, public-only YouTube Data API reads; captions use the existing lane.

Credentials travel in a header, never a URL. Errors expose only allowlisted
reason codes, never server messages, request objects, or response bodies.
"""
import datetime as dt
import html
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

from . import env, health, youtube_yt
from .relevance import token_overlap_relevance

_ENDPOINTS = {'search', 'videos', 'commentThreads'}
_REASONS = {'quotaExceeded', 'dailyLimitExceeded', 'rateLimitExceeded',
            'userRateLimitExceeded', 'keyInvalid', 'accessNotConfigured',
            'forbidden', 'commentsDisabled', 'videoNotFound'}
_MAX_BYTES = 2 * 1024 * 1024


class ApiError(Exception):
    def __init__(self, state, reason):
        self.state = state
        self.reason = reason
        super().__init__(f'YouTube Data API: {reason}')


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward the API key to a redirect target.


def _request(endpoint, params, key, *, deadline=None, cancel=None):
    if endpoint not in _ENDPOINTS:
        raise ApiError(health.ERROR, 'unsupported endpoint')
    if (cancel is not None and cancel.is_set()) or youtube_yt._lane_out_of_time():
        raise ApiError(health.TIMEOUT, 'lane cancelled')
    remaining = 15.0 if deadline is None else min(15.0, deadline - time.monotonic())
    if remaining <= 0:
        raise ApiError(health.TIMEOUT, 'lane deadline')
    url = 'https://www.googleapis.com/youtube/v3/' + endpoint + '?' + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={'X-Goog-Api-Key': key, 'Accept': 'application/json'})
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=remaining) as response:
            payload = response.read(_MAX_BYTES + 1)
        if len(payload) > _MAX_BYTES:
            raise ApiError(health.SCHEMA_DRIFT, 'response too large')
        result = json.loads(payload)
        if not isinstance(result, dict) or not isinstance(result.get('items'), list):
            raise ApiError(health.SCHEMA_DRIFT, 'invalid response schema')
        return result
    except urllib.error.HTTPError as exc:
        reason = f'HTTP {exc.code}'
        try:
            body = json.loads(exc.read(65536))
            candidates = body.get('error', {}).get('errors', [])
            code = candidates[0].get('reason') if candidates else None
            if code in _REASONS:
                reason = code
        except (ValueError, TypeError, AttributeError, IndexError):
            pass
        finally:
            exc.close()
        state = health.ERROR
        if reason in {'quotaExceeded', 'dailyLimitExceeded', 'rateLimitExceeded', 'userRateLimitExceeded'} or exc.code == 429:
            state = health.RATE_LIMITED
        elif exc.code in (401, 403) and reason != 'commentsDisabled':
            state = health.AUTH_FAILED
        raise ApiError(state, reason) from None
    except (TimeoutError, socket.timeout):
        raise ApiError(health.TIMEOUT, 'request timed out') from None
    except (urllib.error.URLError, OSError):
        raise ApiError(health.UNREACHABLE, 'network request failed') from None
    except (ValueError, UnicodeError):
        raise ApiError(health.SCHEMA_DRIFT, 'invalid JSON response') from None


def _obj(value):
    return value if isinstance(value, dict) else {}


def _text(value, limit=5000):
    return html.unescape(value)[:limit] if isinstance(value, str) else ''


def _count(value):
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _item(video_id, snippet, stats=None, details=None):
    snippet, stats, details = _obj(snippet), _obj(stats), _obj(details)
    engagement = {}
    for key, field in [('views', 'viewCount'), ('likes', 'likeCount'), ('comments', 'commentCount')]:
        count = _count(stats.get(field))
        if count is not None:
            engagement[key] = count
    return {
        'video_id': video_id, 'url': f'https://www.youtube.com/watch?v={video_id}',
        'title': _text(snippet.get('title')), 'description': _text(snippet.get('description')),
        'channel_name': _text(snippet.get('channelTitle')), 'channel_id': _text(snippet.get('channelId')),
        'date': _text(snippet.get('publishedAt'), 10) or None,
        'engagement': engagement, 'duration': details.get('duration'),
        'collection_backend': 'youtube-data-api', 'transcript_snippet': '',
        'transcript_highlights': [], 'transcript_status': 'not-requested',
        'top_comments': [], 'comments_status': 'not-requested',
    }


def collect(query, from_date, to_date, depth, config, *, has_ytdlp, freshness_mode=None):
    key = config.get('YOUTUBE_API_KEY')
    if not key:
        return {'items': [], 'error': 'YOUTUBE_API_KEY not configured', 'state': health.SKIPPED_UNCONFIGURED}
    items = []
    deadline = config.get('_research_deadline')
    cancel = config.get('_enrich_cancel')

    def request(endpoint, params):
        return _request(endpoint, params, key, deadline=deadline, cancel=cancel)

    try:
        until = dt.date.fromisoformat(to_date) + dt.timedelta(days=1)
        params = {'part': 'snippet', 'type': 'video', 'q': query, 'order': 'relevance',
                  'maxResults': min(50, youtube_yt.DEPTH_CONFIG.get(depth, 8)),
                  'publishedBefore': until.isoformat() + 'T00:00:00Z'}
        if freshness_mode != 'evergreen_ok':
            params['publishedAfter'] = dt.date.fromisoformat(from_date).isoformat() + 'T00:00:00Z'
        found = request('search', params)
        by_id = {}
        for entry in found['items']:
            entry = _obj(entry)
            video_id = _obj(entry.get('id')).get('videoId')
            if isinstance(video_id, str) and re.fullmatch(r'[A-Za-z0-9_-]{11}', video_id) and video_id not in by_id:
                by_id[video_id] = _item(video_id, entry.get('snippet'))
            if len(by_id) >= params['maxResults']:
                break
        items = list(by_id.values())
        if not items:
            return {'items': []}
        details = request('videos', {'part': 'snippet,statistics,contentDetails', 'id': ','.join(by_id)})
        for entry in details['items']:
            entry = _obj(entry)
            video_id = entry.get('id')
            if isinstance(video_id, str) and video_id in by_id:
                by_id[video_id].update(_item(video_id, entry.get('snippet'), entry.get('statistics'), entry.get('contentDetails')))
        for item in items:
            item['relevance'] = token_overlap_relevance(query, item['title'] + ' ' + item['description'])
            item['why_relevant'] = f"YouTube: {item['title'][:60]}"
        items = [i for i in items if not i['date'] or (
            i['date'] <= to_date and (freshness_mode == 'evergreen_ok' or i['date'] >= from_date))]
        items.sort(key=lambda x: x['engagement'].get('views', 0), reverse=True)
        entity_count = max(1, int(config.get('_comparison_entity_count') or 1))
        comment_limit = youtube_yt.comparison_share(3, entity_count)
        if env.is_youtube_comments_available(config):
            for item in items[:comment_limit]:
                try:
                    response = request('commentThreads', {
                        'part': 'snippet', 'videoId': item['video_id'],
                        'maxResults': 5, 'order': 'relevance', 'textFormat': 'plainText',
                    })
                except ApiError as exc:
                    if exc.reason in {'commentsDisabled', 'videoNotFound'}:
                        item['comments_status'] = 'disabled' if exc.reason == 'commentsDisabled' else 'unavailable'
                        continue
                    item['comments_status'] = exc.state
                    raise
                for entry in response['items'][:5]:
                    comment = _obj(_obj(_obj(entry).get('snippet')).get('topLevelComment'))
                    snippet = _obj(comment.get('snippet'))
                    text = _text(snippet.get('textDisplay'), 2000)
                    if text:
                        item['top_comments'].append({'author': _text(snippet.get('authorDisplayName')),
                            'text': text, 'likes': _count(snippet.get('likeCount')) or 0,
                            'date': _text(snippet.get('publishedAt'), 10),
                            'url': item['url'] + '&lc=' + urllib.parse.quote(str(comment.get('id') or ''), safe='')})
                item['comments_status'] = 'ok'
        else:
            for item in items:
                item['comments_status'] = 'excluded'
        # Data API cannot download arbitrary public captions. Reuse the existing
        # bounded transcript transport, never call captions.list/download.
        target = youtube_yt.comparison_share(youtube_yt.TRANSCRIPT_LIMITS.get(depth, 2), entity_count)
        token = config.get('SCRAPECREATORS_API_KEY') if env.is_youtube_sc_available(config) else None
        if target and (has_ytdlp or token) and not youtube_yt._lane_out_of_time():
            disabled = set()
            transcripts = youtube_yt.fetch_transcripts_parallel(
                [i['video_id'] for i in items[:target]], out_captions_disabled=disabled,
                token=token, max_sc_rescues=target,
            )
            for item in items:
                text = transcripts.get(item['video_id']) or ''
                item['transcript_snippet'] = text
                if item['video_id'] in {i['video_id'] for i in items[:target]}:
                    item['transcript_status'] = 'available' if text else 'unavailable'
                item['transcript_highlights'] = youtube_yt.extract_transcript_highlights(text, query)
                item['captions_disabled'] = item['video_id'] in disabled
        return {'items': items}
    except ApiError as exc:
        return {'items': items, 'error': str(exc), 'state': health.PARTIAL if items else exc.state}
    except (ValueError, TypeError, OverflowError):
        return {'items': items, 'error': 'YouTube Data API: invalid dates or response fields',
                'state': health.PARTIAL if items else health.SCHEMA_DRIFT}
