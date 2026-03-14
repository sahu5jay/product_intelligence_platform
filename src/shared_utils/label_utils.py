import sys
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException


class LabelUtils:
    """
    Utility class for handling Fashion-MNIST labels
    """

    try:

        # Label ID → Label Name
        LABEL_MAP = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot"
        }

        # Reverse Mapping
        LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}

    except Exception as e:
        raise CustomException(e, sys)

    @classmethod
    def get_all_labels(cls):
        """
        Return all labels for dropdown
        """
        try:
            return list(cls.LABEL_TO_ID.keys())

        except Exception as e:
            raise CustomException(e, sys)

    @classmethod
    def get_label_id(cls, label_name: str):
        """
        Convert label name to label ID
        """
        try:

            if label_name not in cls.LABEL_TO_ID:
                raise ValueError(f"Invalid label: {label_name}")

            return cls.LABEL_TO_ID[label_name]

        except Exception as e:
            raise CustomException(e, sys)

    @classmethod
    def get_label_name(cls, label_id: int):
        """
        Convert label ID to label name
        """
        try:

            if label_id not in cls.LABEL_MAP:
                raise ValueError(f"Invalid label id: {label_id}")

            return cls.LABEL_MAP[label_id]

        except Exception as e:
            raise CustomException(e, sys)

    @classmethod
    def validate_label(cls, label_name: str):
        """
        Validate if label exists
        """
        try:

            if label_name not in cls.LABEL_TO_ID:
                raise ValueError(f"Label '{label_name}' is not valid.")

            logging.info(f"Label validated: {label_name}")

            return True

        except Exception as e:
            raise CustomException(e, sys)