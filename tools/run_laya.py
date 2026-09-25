#!/usr/bin/env python3
"""Launch the upstream Laya server with one pinned, CPU-only checkpoint."""
import argparse
import json
import os
from pathlib import Path

MODEL = 'convaiinnovations/laya'
REVISION = '55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851'


def check_budget(ctx):
    """Reject requests that the pinned upstream encoder would silently truncate."""
    from laya.common import render_options, serialize_state
    if ctx.model != 'english' or set(ctx.questions) != {'policy'}:
        raise ValueError('Flowstate requires the English checkpoint and one policy question')
    question = ctx.questions['policy']
    if question.get('type') != 'choice' or set(question.get('criteria', {})) != {
            'cpu_latency', 'gpu_immediate', 'gpu_batch', 'balanced'}:
        raise ValueError('Expected the four Flowstate policies')
    tok = ctx.agent.tok
    count = lambda value: len(tok(value, add_special_tokens=False)['input_ids'])
    internal = {'t': 'choice', 'ins': question['instructions'], 'crit': question['criteria']}
    options = [count(' ' + value) for value in render_options(internal)]
    head = count('choice question: ' + question['instructions']) + sum(options) + len(options)
    if max(options) > 48 or head > ctx.agent.cfg['head_max_len']:
        raise ValueError('Policy descriptions exceed the checkpoint token budget')
    if any(head + count(serialize_state(state)) + 4 > ctx.agent.cfg['max_len'] for state in ctx.states):
        raise ValueError('Telemetry exceeds the checkpoint token budget')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--warmup-request', type=Path)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.threads <= 16:
        parser.error('Port must be 1–65535 and threads 1–16')
    root = Path(__file__).resolve().parents[1]
    os.environ['HF_HOME'] = str(root / '.cache-laya')
    os.environ['LAYA_DEVICE'] = 'cpu'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(MODEL, revision=REVISION, allow_patterns=[
        'rl_agent_config.json', 'model.safetensors', 'tokenizer/*', 'encoder/*'])
    if args.download_only:
        return
    import torch
    import uvicorn
    from laya import Router
    from laya.serve import create_app
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    router = Router(models={'english': model_path}, device='cpu', max_loaded=1,
                    on_predict_start=check_budget)
    router.preload(['english'])
    if args.warmup_request:
        request = json.loads(args.warmup_request.read_text())
        for _ in range(3):
            result = router.predict(request['state'], request['questions'], model='english')
        print(json.dumps({'warmup': result}), flush=True)
    print(json.dumps({'model': MODEL, 'revision': REVISION, 'device': 'cpu',
                      'threads': args.threads, 'port': args.port}), flush=True)
    # Use upstream FastAPI/uvicorn. Limit outstanding calls even if a client times out.
    uvicorn.run(create_app(router), host='127.0.0.1', port=args.port,
                limit_concurrency=2, backlog=8, timeout_keep_alive=2, log_level='warning')


if __name__ == '__main__':
    main()
