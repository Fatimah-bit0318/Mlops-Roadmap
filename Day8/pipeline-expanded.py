from sagemaker.core import ScriptProcessor, image_uris
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.processing import ProcessingInput, ProcessingOutput
from sagemaker.core.shapes.shapes import (
    ProcessingS3Input,
    ProcessingS3Output,
    OutputDataConfig,
    StoppingCondition,
)
from sagemaker.core.training.configs import InputData, Compute
from sagemaker.core.workflow.properties import PropertyFile
from sagemaker.core.workflow.functions import JsonGet
from sagemaker.core.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.train import ModelTrainer
from sagemaker.serve.model_builder import ModelBuilder

REGION = "us-east-1"
ROLE_ARN = "arn:aws:iam::834176180969:role/service-role/AmazonSageMaker-ExecutionRole-20260918T181052"

TRAINING_DATA = "s3://day04/day8/train.csv"
VALIDATION_DATA = "s3://day04/day8/validation.csv"
TEST_DATA = "s3://day04/day8/test.csv"
EVALUATION_SCRIPT = "s3://day04/day8/code/evaluate_model.py"
MODEL_OUTPUT = "s3://day04/day8/model-artifacts/"
EVALUATION_OUTPUT = "s3://day04/day8/evaluation/"
MODEL_PACKAGE_GROUP = "Day8HousePriceModelGroup"
MAX_RMSE = 2_000_000


def main():
    # PipelineSession creates a local definition; it does not start jobs by itself.
    session = PipelineSession()

    xgboost_image = image_uris.retrieve(
        framework="xgboost",
        region=REGION,
        version="3.2-0",
        image_scope="training",
    )

    trainer = ModelTrainer(
        training_image=xgboost_image,
        role=ROLE_ARN,
        sagemaker_session=session,
        base_job_name="day8-house-price",
        compute=Compute(
            instance_type="ml.m5.large",
            instance_count=1,
            volume_size_in_gb=5,
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=600),
        output_data_config=OutputDataConfig(s3_output_path=MODEL_OUTPUT),
        input_data_config=[
            InputData(
                channel_name="train",
                data_source=TRAINING_DATA,
                content_type="text/csv",
            ),
            InputData(
                channel_name="validation",
                data_source=VALIDATION_DATA,
                content_type="text/csv",
            ),
        ],
        hyperparameters={
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "num_round": "10",
            "max_depth": "3",
            "eta": "0.2",
        },
    )

    training_args = trainer.train(wait=False, logs=False)
    training_step = TrainingStep(
        name="TrainHousePriceModel",
        step_args=training_args,
    )

    evaluator = ScriptProcessor(
        image_uri=xgboost_image,
        role=ROLE_ARN,
        command=["python3"],
        instance_count=1,
        instance_type="ml.m5.large",
        volume_size_in_gb=5,
        max_runtime_in_seconds=600,
        base_job_name="day8-house-price-eval",
        sagemaker_session=session,
    )

    evaluation_args = evaluator.run(
        code=EVALUATION_SCRIPT,
        inputs=[
            ProcessingInput(
                input_name="model",
                s3_input=ProcessingS3Input(
                    s3_uri=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                    s3_data_type="S3Prefix",
                    local_path="/opt/ml/processing/model",
                    s3_input_mode="File",
                ),
            ),
            ProcessingInput(
                input_name="test",
                s3_input=ProcessingS3Input(
                    s3_uri=TEST_DATA,
                    s3_data_type="S3Prefix",
                    local_path="/opt/ml/processing/test",
                    s3_input_mode="File",
                ),
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                s3_output=ProcessingS3Output(
                    s3_uri=EVALUATION_OUTPUT,
                    s3_upload_mode="EndOfJob",
                    local_path="/opt/ml/processing/evaluation",
                ),
            )
        ],
        wait=False,
        logs=False,
    )

    evaluation_report = PropertyFile(
        name="HousePriceEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    evaluation_step = ProcessingStep(
        name="EvaluateHousePriceModel",
        step_args=evaluation_args,
        property_files=[evaluation_report],
    )

    inference_image = image_uris.retrieve(
        framework="xgboost",
        region=REGION,
        version="3.2-0",
        image_scope="inference",
    )
    model_builder = ModelBuilder(
        image_uri=inference_image,
        s3_model_data_url=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        role_arn=ROLE_ARN,
        sagemaker_session=session,
    )
    registration_args = model_builder.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=MODEL_PACKAGE_GROUP,
        approval_status="PendingManualApproval",
    )
    register_step = ModelStep(
        name="RegisterHousePriceModel",
        step_args=registration_args,
    )

    rmse_condition = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="regression_metrics.rmse",
        ),
        right=MAX_RMSE,
    )
    quality_gate = ConditionStep(
        name="HousePriceRMSEQualityGate",
        conditions=[rmse_condition],
        if_steps=[register_step],
        else_steps=[],
    )

    pipeline = Pipeline(
        name="Day8-HousePrice-Full-Pipeline",
        steps=[training_step, evaluation_step, quality_gate],
        sagemaker_session=session,
    )

    # This writes the pipeline definition locally; it does NOT create or run it in AWS.
    with open("pipeline-expanded-definition.json", "w") as file:
        file.write(pipeline.definition())

    print("Pipeline definition generated locally.")
    print("No pipeline was created in AWS.")
    print("No training or processing job was started.")


if __name__ == "__main__":
    main()
