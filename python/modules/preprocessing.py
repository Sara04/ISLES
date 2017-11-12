"""Class for ISLES 2017 data preprocessing."""


class PreprocessorISLES(object):
    """Class for ISLES 2017 data preprocessing."""

    """
        Methods:
            get_normalization_parameters: creates a dictionary
                of parameters for volumes' normalization
            get_alignment_parameters: creates a dictionary
                of parameters for volumes' alignment
            name: reproduce PreprocessorISLES object's name
    """
    def get_normalization_parameters(self):
        """Get dictionary of normalization parameters per scan."""
        raise NotImplementedError()

    def get_alignment_parameters(self):
        """Get dictionary of alignment parameters per scan."""
        raise NotImplementedError()

    def name(self):
        """Class name."""
        raise NotImplementedError()
