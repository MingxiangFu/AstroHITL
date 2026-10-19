"""Local Flask UI for category-graph feature mapping."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import torch
from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from hitl_features.images import resolve_image_path
from hitl_features.session import SessionStore, _json_ready


def create_app(sessions_root: Path | str) -> Flask:
    here = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(here / "templates"),
        static_folder=str(here / "static"),
    )
    app.config["STORE"] = SessionStore(Path(sessions_root))

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/sessions")
    def create_session():
        body = request.get_json(force=True) or {}
        try:
            device = body.get("device") or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            session = app.config["STORE"].create(
                feature_path=str(body["feature_path"]),
                pool_size=int(body.get("pool_size", 5000)),
                seed=int(body.get("seed", 0)),
                out_dim=int(body["out_dim"]),
                round_size=int(body.get("round_size", 12)),
                img_dir=str(body.get("img_dir") or ""),
                device=str(device),
                epochs=int(body.get("epochs", 30)),
                pool_indices=body.get("pool_indices"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(snapshot=session.snapshot())

    @app.get("/api/sessions/<session_id>")
    def get_session(session_id: str):
        session = _session_or_404(app, session_id)
        return jsonify(snapshot=session.snapshot())

    @app.post("/api/sessions/<session_id>/recommend")
    def recommend(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(force=True) or {}

        def work(on_progress):
            chosen = session.recommend_round(
                str(body.get("mechanism") or "random"),
                body.get("category_ids"),
            )
            return {"snapshot": session.snapshot(), "chosen": chosen}

        return _respond(work, body)

    @app.post("/api/sessions/<session_id>/place")
    def place(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(force=True) or {}
        try:
            session.place(
                int(body["sample_index"]),
                category_id=body.get("category_id"),
                new_name=body.get("new_name"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(snapshot=session.snapshot())

    @app.post("/api/sessions/<session_id>/unassign")
    def unassign(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(force=True) or {}
        try:
            session.unassign(int(body["sample_index"]))
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(snapshot=session.snapshot())

    @app.post("/api/sessions/<session_id>/recycle")
    def recycle(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(force=True) or {}
        try:
            session.recycle_sample(int(body["sample_index"]))
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(snapshot=session.snapshot())

    @app.post("/api/sessions/<session_id>/position")
    def position(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(force=True) or {}
        try:
            session.set_position(str(body["category_id"]), float(body["x"]), float(body["y"]))
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(snapshot=session.snapshot())

    @app.post("/api/sessions/<session_id>/import-labels")
    def import_labels(session_id: str):
        session = _session_or_404(app, session_id)
        if request.is_json:
            body = request.get_json(force=True) or {}
            text = str(body["csv"]) if "csv" in body else None
            label_column = body.get("label_column")
            index_column = body.get("index_column")
            label_columns = body.get("label_columns")
        else:
            body = {}
            upload = request.files["file"]
            text = upload.read().decode("utf-8-sig")
            label_column = request.form.get("label_column")
            index_column = request.form.get("index_column")
            label_columns = request.form.get("label_columns")

        def work(on_progress):
            if text is None:
                raise KeyError("csv")
            chosen_labels = label_columns
            if isinstance(chosen_labels, str):
                chosen_labels = [part.strip() for part in chosen_labels.split(",") if part.strip()]
            if chosen_labels:
                imported = session.import_hierarchy(
                    text,
                    index_column=str(index_column or ""),
                    label_columns=[str(name) for name in chosen_labels],
                    on_progress=on_progress,
                )
            elif label_column:
                imported = session.import_labels(
                    text, str(index_column), str(label_column), on_progress=on_progress
                )
            else:
                imported = session.import_hierarchy(text, on_progress=on_progress)
            return {"snapshot": session.snapshot(), "imported": imported}

        return _respond(work, body, runtime_status=400)

    @app.get("/api/sessions/<session_id>/labels.csv")
    def export_labels(session_id: str):
        session = _session_or_404(app, session_id)
        return Response(
            session.export_labels(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=labels.csv"},
        )

    @app.post("/api/sessions/<session_id>/ui")
    def update_ui(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(force=True) or {}
        try:
            session.set_ui(body)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(snapshot=session.snapshot())

    @app.post("/api/sessions/<session_id>/view")
    def view(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(force=True) or {}
        try:
            session.enter(body.get("category_id"))
        except KeyError as exc:
            return jsonify(error=str(exc)), 404
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(snapshot=session.snapshot())

    @app.post("/api/sessions/<session_id>/rename")
    def rename(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(force=True) or {}
        try:
            session.rename(str(body["category_id"]), str(body["name"]))
        except (KeyError, ValueError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(snapshot=session.snapshot())

    @app.post("/api/sessions/<session_id>/strength")
    def strength(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(force=True) or {}
        try:
            session.set_strength(str(body["a"]), str(body["b"]), float(body["strength"]))
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(snapshot=session.snapshot())

    @app.post("/api/sessions/<session_id>/confirm")
    def confirm(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(silent=True) or {}

        def work(on_progress):
            row = session.confirm(body.get("epochs"), on_progress=on_progress)
            return {"snapshot": session.snapshot(), "distance": row}

        return _respond(work, body)

    @app.post("/api/sessions/<session_id>/reset-mapper")
    def reset_mapper(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(force=True) or {}
        try:
            session.reset_mapper(bool(body.get("confirm")))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(snapshot=session.snapshot())

    @app.post("/api/sessions/<session_id>/mapper-size")
    def mapper_size(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(force=True) or {}
        try:
            session.set_mapper_hidden(int(body["hidden_dim"]))
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(snapshot=session.snapshot())

    @app.post("/api/sessions/<session_id>/feature-source")
    def feature_source(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(force=True) or {}
        try:
            session.set_feature_source(str(body.get("features") or ""))
        except (TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(snapshot=session.snapshot())

    @app.post("/api/sessions/<session_id>/umap")
    def umap_view(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(silent=True) or {}

        def work(on_progress):
            return {"snapshot": session.snapshot(), "umap": session.umap_layout()}

        return _respond(work, body)

    @app.post("/api/sessions/<session_id>/head-size")
    def head_size(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(force=True) or {}
        try:
            session.set_head_hidden(body["hidden"])
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(snapshot=session.snapshot())

    @app.post("/api/sessions/<session_id>/task-draft")
    def task_draft(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(force=True) or {}
        try:
            session.set_task_draft(body)
        except (KeyError, ValueError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(snapshot=session.snapshot())

    @app.post("/api/sessions/<session_id>/tasks/classification")
    def classification(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(force=True) or {}

        def work(on_progress):
            result = session.train_classification(
                body.get("classes") or [],
                epochs=int(body.get("epochs", 40)),
                fractions=body.get("fractions"),
                on_progress=on_progress,
                features=body.get("features"),
            )
            return {"snapshot": session.snapshot(), "result": result}

        return _respond(work, body, runtime_as_client=True)

    @app.post("/api/sessions/<session_id>/tasks/ranking")
    def ranking(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(force=True) or {}

        def work(on_progress):
            result = session.train_ranking(
                body.get("classes") or [],
                body.get("order") or [],
                float(body.get("strength", 1.0)),
                epochs=int(body.get("epochs", 40)),
                on_progress=on_progress,
                features=body.get("features"),
            )
            return {"snapshot": session.snapshot(), "result": result}

        return _respond(work, body, runtime_as_client=True)

    @app.post("/api/sessions/<session_id>/tasks/select")
    def select_task(session_id: str):
        session = _session_or_404(app, session_id)
        body = request.get_json(force=True) or {}
        try:
            session.show_task(str(body["task_id"]))
        except KeyError as exc:
            return jsonify(error=str(exc)), 404
        return jsonify(snapshot=session.snapshot())

    @app.get("/media/<session_id>/<int:index>")
    def media(session_id: str, index: int):
        session = _session_or_404(app, session_id)
        if not session.img_dir:
            abort(404)
        path = resolve_image_path(session.img_dir, index)
        if path is None:
            abort(404)
        # send_file joins a relative path onto the Flask package directory.
        # The subset form uses subset/images from the directory where the
        # server was started, which is also where the feature file was opened.
        return send_file(Path(path).resolve())

    return app


def _session_or_404(app: Flask, session_id: str):
    try:
        return app.config["STORE"].get(session_id)
    except KeyError:
        abort(404)


def _respond(work, body: dict, *, runtime_status: int = 409, runtime_as_client: bool = False):
    """Return one JSON payload, or newline-delimited progress when ``stream`` is set."""
    if not body.get("stream"):
        try:
            return jsonify(_json_ready(work(None)))
        except RuntimeError as exc:
            status = 400 if runtime_as_client else runtime_status
            return jsonify(error=str(exc)), status
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400
    return _ndjson(work, runtime_status=400 if runtime_as_client else runtime_status)


def _ndjson(work, *, runtime_status: int) -> Response:
    events: queue.Queue = queue.Queue()

    def on_progress(done, total):
        events.put({"event": "progress", "done": int(done), "total": int(total)})

    def runner():
        try:
            payload = work(on_progress)
            events.put({"event": "done", **payload})
        except RuntimeError as exc:
            events.put({"event": "error", "error": str(exc), "status": runtime_status})
        except (KeyError, TypeError, ValueError) as exc:
            events.put({"event": "error", "error": str(exc), "status": 400})
        except Exception as exc:
            events.put({"event": "error", "error": str(exc), "status": 500})
        finally:
            events.put(None)

    threading.Thread(target=runner, daemon=True).start()

    def generate():
        while True:
            item = events.get()
            if item is None:
                break
            yield json.dumps(_json_ready(item)) + "\n"

    return Response(
        generate(),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
