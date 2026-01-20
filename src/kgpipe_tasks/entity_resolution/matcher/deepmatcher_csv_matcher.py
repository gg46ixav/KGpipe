"""
Deepmatcher CSV Matcher task implementation.
"""

from typing import Dict


from kgpipe.common import DataFormat, Data, Registry
from kgpipe.common.io import get_docker_volume_bindings, remap_data_path_for_container
from kgpipe.execution import docker_client

@Registry.task(
    input_spec={"data_dir": DataFormat.DIR, "train": DataFormat.CSV,  "valid": DataFormat.CSV,  "test": DataFormat.CSV,  "unlabeled": DataFormat.CSV},
    output_spec={"best_model": DataFormat.ANY},
    description="DeepMatcher entity matching using Docker container"
)
def deepmatcher_entity_matching(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    DeepMatcher entity matching task that runs in a Docker container.
    
    Args:
        inputs: Dictionary mapping input names to Data objects
        outputs: Dictionary mapping output names to Data objects
    """
    
    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)

    # Extract input paths
    data_dir_path = remap_data_path_for_container(inputs["data_dir"], host_to_container)
    unlabeled_path = remap_data_path_for_container(inputs["unlabeled"], host_to_container)
    best_model_path = remap_data_path_for_container(outputs["best_model"], host_to_container)

    # Ensure output directory exists
    outputs["best_model"].path.parent.mkdir(parents=True, exist_ok=True)

    # Create Docker client with proper volume bindings
    client = docker_client(
        image="kgt/deepmatcher:latest",
        # command=["ls", "-la"],
        command=["bash", "deepmatcher.sh",
                 str(data_dir_path.path),
                 str(inputs["train"].path),
                 str(inputs["valid"].path),
                 str(inputs["test"].path),
                 str(best_model_path.path),
                 str(unlabeled_path.path)],
        volumes=volumes,
    )
    
    # Execute the container
    result = client()
    print(f"DeepMatcher entity matching completed: {result}")
