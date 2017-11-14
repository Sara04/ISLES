"""Class for ISLES 2017 scan/sample info loading and generating."""
import os
import nibabel as nib


class ScanISLES(object):
    """Class for ISLES 2017 scan info loading and generating."""

    def __init__(self, db, name, relative_path, modalities_dict, mode):
        """Initialization of ScanISLES attributes."""
        self.name = name
        self.relative_path = relative_path
        self.modalities_dict = modalities_dict
        self.mode = mode
        self.h = None
        self.w = None
        self.d = None

        _ = self.load_volume(db, 'ADC')

    def load_volume(self, db, m):
        """Loading volume as numpy array."""
        """
            Arguments:
                db: DatabaseISLES
                m: image modality
            Returns:
                volume as numpy array
        """
        volume_path = os.path.join(db.db_path, self.relative_path,
                                   self.modalities_dict[m],
                                   self.modalities_dict[m] + '.nii')

        volume = nib.load(volume_path).get_data().astype('float32')
        self.h, self.w, self.d = volume.shape
        return volume

    def load_volumes(self, db, meta, test=False):
        """Loading all volumes as a list numpy arrays."""
        """
            Arguments:
                db: DatabaseBRATS
            Returns:
                list of volumes
        """
        volumes = [self.load_volume(db, m) for m in db.modalities]
        if not test:
            volumes.append(self.load_volume(db, 'OT'))
        volumes.append(meta.load_brain_mask(db, self))
        return volumes
