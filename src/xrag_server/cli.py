import xrag
from sanic import Sanic
from sanic.request import Request
from sanic.response import json
import argparse

app = Sanic("rag-server")


@app.before_server_start
def start(app):
    app.ctx.ai = xrag.AI()
    app.ctx.store = xrag.Store(app.ctx.ai)


@app.get("/query")
def rag_query(req: Request):
    query = req.args.get("q")
    return json(app.ctx.store.rag(query))


@app.get("/info/<mem_id>")
def rag_info(req: Request, mem_id: str):
    return json(app.ctx.store.info(int(mem_id)))


@app.post("/add")
def rag_add(req: Request):
    if req.form is None:
        return json({"error": "must specify memory in body form"})
    memory = req.form.get("memory")
    if memory is None:
        return json({"error": "must specify memory in body form"})
    return json(app.ctx.store.add([memory]))


@app.post("/relate")
def rag_relate(req: Request):
    child = req.args.get("child")
    parent = req.args.get("parent")
    rel_type = req.args.get("rel_type")
    app.ctx.store.relate(int(child), int(parent), rel_type)
    return json({"success": True})


@app.listener("after_server_stop")
def cleanup_store(app, loop):
    app.ctx.store.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run xrag sanic server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    args = parser.parse_args()

    app.run(host=args.host, port=args.port)
