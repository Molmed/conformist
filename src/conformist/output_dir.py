import os
import shutil
from datetime import datetime


class OutputDir():
    def create_output_dir(self, base_output_dir='output', class_suffix=None):
        # Output directory should be output dir plus class name plus timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        name = self.__class__.__name__
        if class_suffix:
            name = f'{name}_{class_suffix}'

        self.output_dir = os.path.join(base_output_dir,
                                       name,
                                       timestamp)
        os.makedirs(self.output_dir)

    def export(self, filename, dst):
        src = f'{self.output_dir}/{filename}'
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)
        print(f"Exported to {dst}/{filename}")
