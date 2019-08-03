from setuptools import setup, find_packages

setup(
    name='bevutils',
    version='1.0.0',
    url='https://github.com/huangyuyao/bevutils.git',
    author='Huang Yuyao, Wen Yongkun',
    author_email='huangyuyao@outlook.com',
    description='Bird\'s-Eye View Utilities',
    packages=find_packages(),
    install_requires=[
        'numpy >= 1.11.1',
        'matplotlib >= 1.5.1',
        'opencv-python',
        'easydict'
    ],
)


