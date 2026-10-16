"""Start or cancel an external `v2vmd run` without blocking the UI."""

from __future__ import annotations

from pathlib import Path

import bpy
from bpy.types import Context, Operator

from ..i18n import t
from ..preferences import get_preferences
from ..registration import safe_register_class, safe_unregister_class
from ..services.backend import (
    build_run_argv,
    job_output_dir,
    launch_process,
    new_job_id,
    subprocess_env,
    validate_run_inputs,
)
from .runtime import JobRuntime, attach, is_busy, request_cancel


class V2VMD_OT_run_pipeline(Operator):
    bl_idname = "v2vmd.run_pipeline"
    bl_label = "Generate Motion"
    bl_description = "Run Video2VMD in the configured External Python"

    def execute(self, context: Context):
        prefs = get_preferences(context)
        props = context.scene.v2vmd
        source_type = props.source_type.lower()
        errors = validate_run_inputs(
            python=prefs.python_path,
            project_dir=prefs.project_dir,
            source=props.source_path,
            model=props.model_path,
            busy=is_busy(),
        )
        if errors:
            props.job_status = "failed"
            props.job_error = errors[0]
            self.report({"ERROR"}, t(errors[0]))
            return {"CANCELLED"}
        if props.output_format == "vmd" and props.ik_mode == "ik" and not props.model_path.strip():
            self.report({"ERROR"}, t("err_model"))
            return {"CANCELLED"}

        source = Path(props.source_path).expanduser()
        job_id = new_job_id(source.stem)
        output_dir = job_output_dir(prefs.project_dir, job_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_path = output_dir / "progress.jsonl"
        log_path = output_dir / "job.log"

        argv = build_run_argv(
            prefs.python_path,
            source_type=source_type,
            source=source,
            model=props.model_path,
            output_format=props.output_format,
            foot_lock=bool(props.foot_lock),
            ik_mode=props.ik_mode if props.output_format == "vmd" else "fk",
            device=props.device,
            output_dir=output_dir,
            progress_jsonl=progress_path,
            preview=bool(props.generate_preview),
        )
        try:
            process, log_file = launch_process(
                argv,
                cwd=prefs.project_dir,
                log_path=log_path,
                env=subprocess_env(prefs.project_dir),
            )
        except OSError as exc:
            props.job_status = "failed"
            props.job_error = f"err_python: {exc}"
            self.report({"ERROR"}, f"{t('err_python')}: {exc}")
            return {"CANCELLED"}

        props.job_status = "running"
        props.job_stage = "preflight"
        props.job_progress = 0.05
        props.job_error = ""
        props.import_note = ""
        props.job_motion_path = ""
        props.job_output_dir = str(output_dir)
        props.job_log_path = str(log_path)
        props.job_id = job_id
        props.job_source_name = source.stem

        attach(
            JobRuntime(
                kind="run",
                process=process,
                owned_pid=process.pid,
                output_dir=output_dir,
                log_path=log_path,
                progress_path=progress_path,
                log_file=log_file,
                source_name=source.stem,
                auto_import=props.output_format == "bvh",
            )
        )
        self.report({"INFO"}, t("info_started_job", job=job_id))
        return {"FINISHED"}


class V2VMD_OT_cancel_job(Operator):
    bl_idname = "v2vmd.cancel_job"
    bl_label = "Cancel"
    bl_description = "Stop the Video2VMD job started by this add-on"

    def execute(self, context: Context):
        if not request_cancel():
            self.report({"WARNING"}, t("err_no_cancel"))
            return {"CANCELLED"}
        context.scene.v2vmd.job_status = "cancelling"
        context.scene.v2vmd.job_error = "cancelling_ellipsis"
        self.report({"INFO"}, t("info_cancelling"))
        return {"FINISHED"}


def register() -> None:
    safe_register_class(V2VMD_OT_run_pipeline)
    safe_register_class(V2VMD_OT_cancel_job)


def unregister() -> None:
    safe_unregister_class(V2VMD_OT_cancel_job)
    safe_unregister_class(V2VMD_OT_run_pipeline)
