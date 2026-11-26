"""Admin CLI: bootstrap, embedding indexing, health diagnostics."""
import argparse
import json
from .auth import init_db
from .documents import init_docs,seed_documents,refresh_embeddings,list_docs
from .ollama import ready

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['bootstrap','reindex','status'])
    p.add_argument('--no-embed',action='store_true')
    args=p.parse_args()
    init_db();init_docs()
    if args.command=='status':
        print(json.dumps({'ollama':ready(),'documents':len(list_docs({'role':'admin'}))},ensure_ascii=False,indent=2));return
    if args.command=='bootstrap':
        print(f"Seed documents added: {seed_documents()}",flush=True)
    if not args.no_embed:
        print('Indexing multilingual embeddings...',flush=True)
        print(json.dumps(refresh_embeddings(),ensure_ascii=False),flush=True)

if __name__=='__main__': main()
