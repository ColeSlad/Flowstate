#!/usr/bin/env python3
"""Exercise the real HTTP/SSE server, bounded clients, controls, and shutdown."""
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

gpu = '--gpu' in sys.argv
with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    port = probe.getsockname()[1]
base = f'http://127.0.0.1:{port}'
process = subprocess.Popen([sys.argv[1], '--port', str(port), '--vectors', '1000',
                            '--dimension', '384' if gpu else '9', '--cuda', 'on' if gpu else 'off'],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
streams = []


def request(path, data=None, headers=None):
    body = json.dumps(data).encode() if data is not None else None
    return urllib.request.urlopen(urllib.request.Request(base + path, data=body,
        headers=headers or ({'Content-Type': 'application/json'} if body else {})), timeout=3)


def state():
    with request('/api/state') as response:
        return json.load(response)


def until(predicate, timeout=8):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if process.poll() is not None:
            output = process.communicate()
            if gpu and 'CUDA unavailable' in output[1]:
                print('CUDA unavailable: skipping dashboard GPU test')
                raise SystemExit(77)
            raise AssertionError('Server exited early: ' + repr(output))
        try:
            result = predicate()
            if result:
                return result
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(.05)
    raise AssertionError('Timed out waiting for server condition')


def fails(status, path='/api/controls', data=None, headers=None):
    try:
        request(path, data, headers)
    except urllib.error.HTTPError as error:
        assert error.code == status, (error.code, status)
    else:
        raise AssertionError(f'Expected HTTP {status}')


try:
    initial = until(state, timeout=15)
    assert initial['gpu_available'] == gpu
    assert initial['controls'] == {'traffic': 'low', 'top_k': 10, 'contention': False}
    if not gpu:
        assert initial['gpu_utilization'] is None and initial['gpu_memory_free_bytes'] is None
    for path, fragment in [('/', 'Work follows load.'), ('/app.js', 'EventSource'), ('/style.css', 'prefers-reduced-motion')]:
        with request(path) as response:
            assert fragment in response.read().decode()
    with request('/api/comparison') as response:
        assert len(json.load(response)['modes']) == 5
    valid = {'traffic': 'burst', 'top_k': 50, 'contention': False}
    for change in [{'traffic': 'invalid'}, {'top_k': 0}, {'top_k': -1}, {'top_k': 51},
                   {'top_k': 1.5}, {'contention': 'on'}, {'extra': 1}]:
        fails(400, data={**valid, **change})
    fails(415, data=valid, headers={'Content-Type': 'text/plain'})
    fails(403, data=valid, headers={'Content-Type': 'application/json', 'Origin': 'https://unrelated.example'})
    fails(403, path='/api/state', headers={'Host': 'unrelated.example'})
    fails(413, data={'traffic': 'x' * 3000})
    fails(400, data={'traffic': [[[[[[1]]]]]]})
    with request('/api/controls', valid) as response:
        assert json.load(response) == valid
    until(lambda: state()['arrival_rate'] > 100)
    loaded = state()
    assert loaded['controls'] == valid and loaded['total_completed'] > 0
    if gpu:
        with request('/api/controls', {**valid, 'contention': True}):
            pass
        until(lambda: state()['contention']['kernels'] > 2)
        busy = state()
        assert busy['contention']['active'] and not busy['contention']['error']
        with request('/api/controls', valid):
            pass
        until(lambda: not state()['contention']['active'])
    else:
        fails(400, data={**valid, 'contention': True})
    # Each stream reserves one of four slots. Controls and snapshots remain usable.
    for _ in range(4):
        stream = request('/events')
        assert stream.headers['Content-Type'].startswith('text/event-stream')
        line = stream.readline().decode()
        assert line.startswith('data: ') and json.loads(line[6:])['vectors'] == 1000
        streams.append(stream)
    fails(503, path='/events')
    assert state()['total_completed'] >= loaded['total_completed']
    for stream in streams:
        stream.close()
    streams.clear()
    # Broken clients release their slots; a fresh connection must eventually succeed.
    reopened = until(lambda: request('/events'))
    streams.append(reopened)
    assert reopened.readline().startswith(b'data: ')
    process.terminate()  # Includes an active stream and load, not just an idle server.
    stdout, stderr = process.communicate(timeout=12)
    assert process.returncode == 0, (stdout, stderr)
    final = json.loads(stdout.strip().splitlines()[-1])
    assert final['submitted'] == final['completed'] and final['failed'] == 0, final
    print('Dashboard HTTP/SSE/control/shutdown checks passed' + (' with real CUDA contention' if gpu else ''))
finally:
    for stream in streams:
        stream.close()
    if process.poll() is None:
        process.kill()
        process.communicate(timeout=5)
