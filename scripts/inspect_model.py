from torchview import draw_graph
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gen2act.modeling import build_default_policy
from train_policy import load_config


config = load_config(REPO_ROOT / "configs" / "gen2act_policy_toto_gen.toml")
model_cfg = config["model"]
data_cfg = config["data"]
train_cfg = config["train"]
track_cfg = config.get("track", {})
wandb_cfg = config.get("wandb", {})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_default_policy(
    num_action_dims=int(model_cfg["num_action_dims"]),
    num_bins=int(model_cfg["num_bins"]),
    image_size=int(model_cfg["image_size"]),
    patch_size=int(model_cfg["patch_size"]),
    hidden_dim=int(model_cfg["hidden_dim"]),
    num_vit_layers=int(model_cfg["num_vit_layers"]),
    num_vit_heads=int(model_cfg["num_vit_heads"]),
    latent_tokens=int(model_cfg["latent_tokens"]),
    human_video_len=int(data_cfg["human_video_len"]),
    robot_history_len=int(data_cfg["robot_history_len"]),
    vit_pretrained=model_cfg.get("pretrained"),
    enable_point_tracking=bool(track_cfg.get("enable", False)),
    point_tracker_checkpoint=None,
    point_tracker_use_hub=bool(track_cfg.get("use_hub", False)),
    point_tracker_offline=bool(track_cfg.get("offline", True)),
    point_tracker_v2=bool(track_cfg.get("use_v2_model", False)),
    point_tracker_window_len=int(track_cfg.get("window_len", 60)),
    track_grid_size=int(track_cfg.get("grid_size", 10)),
    track_query_frame=int(track_cfg.get("grid_query_frame", 0)),
    track_backward=bool(track_cfg.get("backward_tracking", False)),
).to(device)

mock_human_video = torch.randn(1, 16, 3, 224, 224)    # [B, T, C, H, W] 人类演示视频
mock_robot_history = torch.randn(1, 8, 3, 224, 224)             # [B, T, joint_dim] 机器人历史状态/动作
mock_human_tracks = torch.randn(1, 100, 16, 2)             # [B, N, T, 2] 提取的人类关键点轨迹
mock_robot_tracks = torch.randn(1, 100, 8, 2)             # [B, N, T, 2] 机器人末端轨迹真值

input_data = (
    None,               # scene_img
    None,               # task_prompt_tokens
    mock_human_video,
    mock_robot_history,
    mock_human_tracks,
    mock_robot_tracks,
    False               # debug_isfinite
)

model_graph = draw_graph(
    model, 
    input_data=input_data,   # 直接作为位置参数传给 input_data
    device=device, 
    depth=3,
    expand_nested=True,
    show_shapes=True
)

# 5. 渲染并保存为图片
# 这会在当前目录下生成一个 "robot_pipeline_architecture.png" 文件
model_graph.visual_graph.render("robot_pipeline_architecture", format="png")
