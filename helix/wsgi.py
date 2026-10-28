"""Production WSGI entrypoint. Hosted deployment requires explicit HTTP Basic Auth credentials."""
import os

from helix.web import create_app

app = create_app(
    config_path=os.getenv("HELIX_CONFIG_PATH", "configs/helix-deploy.json"),
    llm_checkpoint=os.getenv("HELIX_CHECKPOINT_PATH", "artifacts/sft/sckpt-best.pt"),
    image_gen_config=os.getenv("HELIX_IMAGE_CONFIG_PATH", "artifacts/image-gen/config.json"),
    image_gen_checkpoint=os.getenv("HELIX_IMAGE_CHECKPOINT_PATH", "artifacts/image-gen/ckpt-final.pt"),
    vision_checkpoint=os.getenv("HELIX_VISION_CHECKPOINT_PATH", "artifacts/vision/ckpt-final.pt"),
    caption_dir=os.getenv("HELIX_CAPTION_DIR", "artifacts/caption"),
    external_origin=os.getenv("HELIX_EXTERNAL_ORIGIN") or None,
)
