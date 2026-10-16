"""Import BVH skeletons or apply VMD to the active PMX armature via MMD Tools."""
from pathlib import Path

import bpy
from bpy.types import Context, Operator

from ..i18n import t
from ..registration import safe_register_class, safe_unregister_class
from ..services.backend import collection_name_for_source
from ..services.result_reader import RunResult, blender_frame_settings, read_run_json

def _view3d_region(area):
    for region in area.regions:
        if region.type == "WINDOW":
            return region
    return None

def _frame_selected(context: Context) -> None:
    wm = context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = _view3d_region(area)
            if region is None:
                continue
            try:
                with context.temp_override(window=window, area=area, region=region):
                    bpy.ops.view3d.view_selected()
                return
            except Exception:
                return

def apply_scene_frames(scene, fps: float | None, n_frames: int | None) -> None:
    if fps is None or n_frames is None:
        return
    settings = blender_frame_settings(fps, n_frames)
    scene.render.fps = int(settings["fps"])
    scene.render.fps_base = float(settings["fps_base"])
    scene.frame_start = int(settings["frame_start"])
    scene.frame_end = int(settings["frame_end"])


def import_generated_motion(context: Context, result: RunResult, *, source_name: str) -> None:
    path = result.motion_path
    if path is None or not path.is_file():
        raise FileNotFoundError("Result motion file not found")
    if path.suffix.lower() == ".bvh":
        try:
            bpy.ops.import_anim.bvh.get_rna_type()
        except (AttributeError, KeyError, RuntimeError):
            try:
                import io_anim_bvh
                io_anim_bvh.register()
            except ImportError as exc:
                raise RuntimeError("Enable Blender's BVH importer before importing motion") from exc
        existing = set(bpy.data.objects)
        status = bpy.ops.import_anim.bvh(filepath=str(path), global_scale=1., frame_start=1,
                                         axis_forward="-Z", axis_up="Y", rotate_mode="QUATERNION")
        if status != {"FINISHED"}:
            raise RuntimeError("BVH import did not complete")
        objects = [obj for obj in bpy.data.objects if obj not in existing]
        collection = bpy.data.collections.new(collection_name_for_source(source_name))
        context.scene.collection.children.link(collection)
        for obj in objects:
            for owner in list(obj.users_collection):
                owner.objects.unlink(obj)
            collection.objects.link(obj)
        arm = next((obj for obj in objects if obj.type == "ARMATURE"), None)
        if arm is not None:
            arm.show_in_front = True
            arm.data.display_type = "STICK"
            arm.select_set(True)
            context.view_layer.objects.active = arm
        _frame_selected(context)
    elif path.suffix.lower() == ".vmd":
        arm = context.view_layer.objects.active
        if arm is None or arm.type != "ARMATURE":
            raise RuntimeError("Select the target PMX armature before importing VMD")
        try:
            bpy.ops.mmd_tools.import_vmd.get_rna_type()
        except (AttributeError, KeyError, RuntimeError) as exc:
            raise RuntimeError("Enable MMD Tools to import VMD onto the selected PMX armature") from exc
        # MMD Tools imports onto every selected object. Scope this operation
        # to the active character even if other armatures are selected.
        with context.temp_override(active_object=arm, object=arm, selected_objects=[arm], selected_editable_objects=[arm]):
            status = bpy.ops.mmd_tools.import_vmd(filepath=str(path), scale=.085, margin=1,
                                                 bone_mapper="PMX", use_pose_mode=False)
        if status != {"FINISHED"}:
            raise RuntimeError("VMD import did not complete")
    else:
        raise ValueError(f"Unsupported motion format: {path.suffix}")
    apply_scene_frames(context.scene, result.fps, result.n_frames)
    context.scene.frame_set(1)


class V2VMD_OT_import_result(Operator):
    bl_idname = "v2vmd.import_result"
    bl_label = "Import Motion"
    bl_description = "Import the last BVH, or apply the last VMD to the active PMX armature"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: Context):
        props = context.scene.v2vmd
        if not props.job_output_dir:
            self.report({"ERROR"}, t("err_no_job_folder"))
            return {"CANCELLED"}
        result = read_run_json(Path(props.job_output_dir).expanduser())
        try:
            import_generated_motion(context, result, source_name=props.job_source_name or result.run_dir.name)
        except Exception as exc:
            props.import_note = "generation_import_failed"
            props.job_error = f"generation_import_failed: {exc}"
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        props.job_motion_path = str(result.motion_path)
        props.import_note = ""
        props.job_error = ""
        props.job_status = "completed"
        self.report({"INFO"}, t("info_imported", name=result.motion_path.name))
        return {"FINISHED"}


def register() -> None:
    safe_register_class(V2VMD_OT_import_result)


def unregister() -> None:
    safe_unregister_class(V2VMD_OT_import_result)
