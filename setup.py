from setuptools import setup, find_packages

setup(
    name='bevutils',
    version='1.1.1',
    url='https://github.com/huangyuyao/bevutils.git',
    author='Huang Yuyao, Wen Yongkun',
    author_email='huangyuyao@outlook.com',
    description='Bird\'s-Eye View Utilities',
    packages=find_packages(),
    install_requires=[
        'numpy == 1.16.0',
        'matplotlib >= 3.0.1',
        'opencv-python >= 4.1.0.25',
        'easydict >= 1.9',
        'torch >= 1.1.0',
        'torchvision >= 0.3.0',
        'logger >= 1.4',
        'tensorboard >= 1.14.0',
        'tensorboardx',
    ],
    entry_points={
        'console_scripts' : [
            'bevtrain = bevutils.trainer.train:main',
        ],
    }
)


