
import os
import sys
import zipfile
import shutil
from pathlib import Path

import yaml
from cellSegmentation.utils.main_utils import read_yaml_file
from cellSegmentation.logger import logging
from cellSegmentation.exception import AppException
from cellSegmentation.entity.config_entity import ModelTrainerConfig
from cellSegmentation.entity.artifacts_entity import ModelTrainerArtifact


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Starting YOLO training...")

            run_name = self.model_trainer_config.run_name

            # Train YOLOv8 with dynamic run name
            os.system(
                f"yolo task=segment mode=train model={self.model_trainer_config.weight_name} "
                f"data=data.yaml epochs={self.model_trainer_config.no_epochs} imgsz=640 "
                f"save=true name={run_name}"
            )

            # Dynamically constructed model path
            output_dir = os.path.join("runs", "segment", run_name)
            src_model = os.path.join(output_dir, "weights", "best.pt")
            dest_dir = self.model_trainer_config.model_trainer_dir
            dest_model = os.path.join(dest_dir, "best.pt")

            if not os.path.exists(src_model):
                raise FileNotFoundError(f"Expected trained model at {src_model}, but not found!")

            os.makedirs(dest_dir, exist_ok=True)
            os.replace(src_model, dest_model)

            # Clean up artifacts
            if os.path.exists("runs"):
                shutil.rmtree("runs")

            if os.path.exists("yolov8s-seg.pt"):
                os.remove("yolov8s-seg.pt")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=dest_model
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise AppException(e, sys)


"""
    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data.zip...")
            # Unzip the data
            with zipfile.ZipFile("data.zip", 'r') as zip_ref:
                zip_ref.extractall()
            os.remove("data.zip")

            # Run YOLO training
            logging.info("Starting YOLO training...")
            train_command = (
                f"yolo task=segment mode=train model={self.model_trainer_config.weight_name} "
                f"data=data.yaml epochs={self.model_trainer_config.no_epochs} imgsz=640 save=true"
            )
            exit_code = os.system(train_command)
            if exit_code != 0:
                raise Exception("YOLO training failed. Check your model path or data.yaml.")

            # Create output directory
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)

            # Copy the trained model (best.pt)
            src_model = Path("runs/segment/train/weights/best.pt")
            dst_model = Path(self.model_trainer_config.model_trainer_dir) / "best.pt"
            if src_model.exists():
                shutil.copy(src_model, dst_model)
            else:
                raise FileNotFoundError(f"Expected trained model at {src_model}, but not found!")

            # Clean up unnecessary files and folders
            for item in [
                "yolov8s-seg.pt", "train", "valid", "test", "data.yaml", "runs"
            ]:
                path = Path(item)
                if path.is_file():
                    os.remove(path)
                elif path.is_dir():
                    shutil.rmtree(path)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=str(dst_model),
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise AppException(e, sys)
"""
        
