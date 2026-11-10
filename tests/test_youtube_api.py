"""Official YouTube public API contract. All network is mocked."""
import io
import json
import time
import urllib.error
from unittest import mock

import pytest
from lib import env, health, pipeline, backends, normalize, youtube_api

KEY = 'not-a-real-test-key'
VIDEO = 'abcdefghijk'
SEARCH = {'items': [{'id': {'videoId': VIDEO}, 'snippet': {
    'title': 'Agent &amp; tools', 'channelTitle': 'Channel', 'channelId': 'UCtest',
    'publishedAt': '2026-11-05T12:00:00Z', 'description': 'A review'}}]}
DETAILS = {'items': [{'id': VIDEO, 'snippet': SEARCH['items'][0]['snippet'],
                     'statistics': {'viewCount': '100', 'likeCount': '5'},
                     'contentDetails': {'duration': 'PT5M'}}]}
COMMENTS = {'items': [{'snippet': {'topLevelComment': {'id': 'comment1', 'snippet': {
    'authorDisplayName': 'Viewer', 'textDisplay': 'Useful test', 'likeCount': 2,
    'publishedAt': '2026-11-06T00:00:00Z'}}}}]}
CONFIG = {'YOUTUBE_API_KEY': KEY}


def collect(config=None, depth='quick', **kwargs):
    return youtube_api.collect('agents', '2026-11-01', '2026-11-10', depth,
                               CONFIG if config is None else config, has_ytdlp=False, **kwargs)


def test_no_key_is_offline():
    with mock.patch.object(youtube_api, '_request') as request:
        result = collect({})
    request.assert_not_called()
    assert result['items'] == []
    assert result['state'] == 'skipped-unconfigured'


def test_public_discovery_metadata_comments_and_inclusive_dates():
    with mock.patch.object(youtube_api, '_request', side_effect=[SEARCH, DETAILS, COMMENTS]) as request:
        result = collect()
    assert 'error' not in result
    item = result['items'][0]
    assert item['title'] == 'Agent & tools'
    assert item['engagement'] == {'views': 100, 'likes': 5}
    assert item['date'] == '2026-11-05'
    assert item['top_comments'][0]['text'] == 'Useful test'
    assert item['collection_backend'] == 'youtube-data-api'
    assert item['transcript_snippet'] == ''
    calls = request.call_args_list
    assert [c.args[0] for c in calls] == ['search', 'videos', 'commentThreads']
    params = calls[0].args[1]
    assert params['publishedAfter'] == '2026-11-01T00:00:00Z'
    assert params['publishedBefore'] == '2026-11-11T00:00:00Z'
    assert params['type'] == 'video'
    assert calls[2].args[1]['textFormat'] == 'plainText'


def test_empty_success_never_calls_enrichment():
    with mock.patch.object(youtube_api, '_request', return_value={'items': []}) as request:
        result = collect()
    assert result == {'items': []}
    assert request.call_count == 1


def test_metadata_failure_preserves_search_and_marks_partial():
    with mock.patch.object(youtube_api, '_request', side_effect=[SEARCH, youtube_api.ApiError(health.RATE_LIMITED, 'quotaExceeded')]):
        result = collect()
    assert result['items'][0]['engagement'] == {}
    assert result['state'] == health.PARTIAL
    assert 'quotaExceeded' in result['error']


def test_comments_disabled_is_explicit_not_failure():
    with mock.patch.object(youtube_api, '_request', side_effect=[SEARCH, DETAILS, youtube_api.ApiError(health.ERROR, 'commentsDisabled')]):
        result = collect()
    assert 'error' not in result
    assert result['items'][0]['comments_status'] == 'disabled'


def test_comments_failure_partial_and_opt_out():
    with mock.patch.object(youtube_api, '_request', side_effect=[SEARCH, DETAILS, youtube_api.ApiError(health.AUTH_FAILED, 'forbidden')]):
        result = collect()
    assert result['state'] == health.PARTIAL
    with mock.patch.object(youtube_api, '_request', side_effect=[SEARCH, DETAILS]) as request:
        result = collect({**CONFIG, 'EXCLUDE_SOURCES': 'youtube_comments'})
    assert request.call_count == 2
    assert result['items'][0]['comments_status'] == 'excluded'


def test_duplicates_and_invalid_ids_do_not_create_requests():
    data = {'items': SEARCH['items'] * 2 + [{'id': {'videoId': '../bad'}}]}
    with mock.patch.object(youtube_api, '_request', side_effect=[data, DETAILS, COMMENTS]):
        result = collect()
    assert len(result['items']) == 1


def test_evergreen_omits_lower_date_bound_only():
    with mock.patch.object(youtube_api, '_request', return_value={'items': []}) as request:
        collect(freshness_mode='evergreen_ok')
    params = request.call_args.args[1]
    assert 'publishedAfter' not in params
    assert 'publishedBefore' in params


def test_transport_sends_key_in_header_not_url_and_suppresses_server_text():
    opener = mock.Mock()
    opener.open.side_effect = urllib.error.HTTPError('https://example.test/?key=' + KEY, 403,
        KEY, {}, io.BytesIO(json.dumps({'error': {'message': KEY, 'errors': [{'reason': 'quotaExceeded'}]}}).encode()))
    with mock.patch.object(youtube_api.urllib.request, 'build_opener', return_value=opener):
        with pytest.raises(youtube_api.ApiError) as caught:
            youtube_api._request('search', {'q': 'agents'}, KEY)
    assert caught.value.state == health.RATE_LIMITED
    assert KEY not in str(caught.value)
    req = opener.open.call_args.args[0]
    assert KEY not in req.full_url
    assert req.get_header('X-goog-api-key') == KEY


def test_deadline_and_unsupported_endpoint_are_offline():
    with mock.patch.object(youtube_api.urllib.request, 'build_opener') as opener:
        with pytest.raises(youtube_api.ApiError):
            youtube_api._request('search', {}, KEY, deadline=time.monotonic()-1)
        with pytest.raises(youtube_api.ApiError):
            youtube_api._request('https://evil.test/', {}, KEY)
    opener.assert_not_called()


def test_transcript_path_is_separate_and_bounded():
    with mock.patch.object(youtube_api, '_request', side_effect=[SEARCH, DETAILS, COMMENTS]), \
         mock.patch.object(youtube_api.youtube_yt, 'fetch_transcripts_parallel', return_value={VIDEO: 'Transcript words'}) as fetch:
        result = youtube_api.collect('agents', '2026-11-01', '2026-11-10', 'default', CONFIG, has_ytdlp=True)
    assert fetch.call_args.args[0] == [VIDEO]
    assert result['items'][0]['transcript_snippet'] == 'Transcript words'


def test_key_configuration_and_source_gates(monkeypatch):
    monkeypatch.setenv('YOUTUBE_API_KEY', KEY)
    assert env.get_config()['YOUTUBE_API_KEY'] == KEY
    with mock.patch.object(pipeline, 'which', return_value=None):
        assert 'youtube' in pipeline.available_sources(CONFIG)
    assert env.is_youtube_api_available(CONFIG)
    assert not env.is_youtube_api_available({**CONFIG, 'LAST30DAYS_YOUTUBE_BACKEND': 'yt-dlp'})
    assert backends.resolve('youtube', CONFIG).active_backend == 'youtube-api'


def test_pipeline_calls_official_backend_and_preserves_errors():
    with mock.patch.object(youtube_api, 'collect', return_value={'items': [], 'state': health.RATE_LIMITED, 'error': 'quotaExceeded'}) as collect_mock, \
         mock.patch.object(pipeline.youtube_yt, 'search_and_transcribe') as old:
        items, artifact = pipeline._fetch_youtube_stream('agents', '2026-11-01', '2026-11-10', 'quick', CONFIG, has_ytdlp=True)
    collect_mock.assert_called_once()
    old.assert_not_called()
    assert not items
    assert 'quotaExceeded' in str(artifact)


def test_provenance_survives_normalization():
    with mock.patch.object(youtube_api, '_request', side_effect=[SEARCH, DETAILS, COMMENTS]):
        item = collect()['items'][0]
    normalized = normalize._normalize_youtube('youtube', item, 0, '2026-11-01', '2026-11-10')
    assert normalized.metadata['collection_backend'] == 'youtube-data-api'
    assert normalized.metadata['channel_id'] == 'UCtest'
    assert normalized.metadata['comments_status'] == 'ok'


def test_sc_transcripts_remain_available_without_ytdlp():
    with mock.patch.object(youtube_api, '_request', side_effect=[SEARCH, DETAILS, COMMENTS]), \
         mock.patch.object(youtube_api.youtube_yt, 'fetch_transcripts_parallel', return_value={VIDEO: 'Words'}) as fetch:
        result = collect({**CONFIG, 'SCRAPECREATORS_API_KEY': 'fake-sc'}, depth='default')
    assert fetch.call_args.kwargs['token'] == 'fake-sc'
    assert result['items'][0]['transcript_status'] == 'available'


def test_http_untrusted_reason_cannot_escape_as_error():
    opener = mock.Mock()
    opener.open.side_effect = urllib.error.HTTPError('url', 403, KEY, {},
        io.BytesIO(json.dumps({'error': {'errors': [{'reason': KEY}]}}).encode()))
    with mock.patch.object(youtube_api.urllib.request, 'build_opener', return_value=opener):
        with pytest.raises(youtube_api.ApiError) as caught:
            youtube_api._request('search', {}, KEY)
    assert KEY not in str(caught.value)
    assert caught.value.state == health.AUTH_FAILED


def test_response_size_schema_and_redirect_guard():
    for payload in [b'x' * (youtube_api._MAX_BYTES+1), b'{bad', b'{"error": "private"}']:
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = payload
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(youtube_api.urllib.request, 'build_opener', return_value=opener):
            with pytest.raises(youtube_api.ApiError) as caught:
                youtube_api._request('search', {}, KEY)
        assert caught.value.state == health.SCHEMA_DRIFT
    assert youtube_api._NoRedirect().redirect_request(None,None,302,'',{},'https://evil.test') is None


def test_cancel_prevents_all_network_calls():
    cancel = mock.Mock()
    cancel.is_set.return_value = True
    with mock.patch.object(youtube_api.urllib.request, 'build_opener') as opener:
        with pytest.raises(youtube_api.ApiError):
            youtube_api._request('search', {}, KEY, cancel=cancel)
    opener.assert_not_called()


def test_end_date_boundary_is_filtered_after_metadata():
    details = json.loads(json.dumps(DETAILS))
    details['items'][0]['snippet']['publishedAt'] = '2026-11-11T00:00:00Z'
    with mock.patch.object(youtube_api, '_request', side_effect=[SEARCH, details]) as request:
        result = collect()
    assert result['items'] == []
    assert request.call_count == 2


def test_later_transcript_backfill_updates_provenance():
    from lib import youtube_yt
    item = normalize._normalize_youtube('youtube', {
        'video_id': VIDEO, 'collection_backend': 'youtube-data-api',
        'transcript_status': 'unavailable'}, 0, '2026-11-01', '2026-11-10')
    with mock.patch.object(youtube_yt, 'is_ytdlp_installed', return_value=True), \
         mock.patch.object(youtube_yt, 'fetch_transcripts_parallel', return_value={VIDEO: 'Transcript words'}):
        youtube_yt.backfill_transcripts([item], topic='agents')
    assert item.metadata['transcript_status'] == 'available'


def test_search_and_comment_budget_remains_bounded():
    found = {'items': [{**SEARCH['items'][0], 'id': {'videoId': f'{i:011d}'}} for i in range(20)]}
    with mock.patch.object(youtube_api, '_request', side_effect=[found, {'items': []}, COMMENTS, COMMENTS, COMMENTS]) as request:
        result = collect()
    assert len(result['items']) == 6
    assert request.call_count == 5
    assert sum(len(i['top_comments']) for i in result['items']) == 3
