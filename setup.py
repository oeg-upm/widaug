

from pathlib import Path

from setuptools import find_packages, setup

# requirements = Path(__file__).parent/ 'requirements.txt'

# with requirements.open(mode='rt', encoding='utf-8') as fp:
#    install_requires = [line.strip() for line in fp]

setup(
    name='widaug',
    version='0.0.1',
    description='Data Augmentation Using Wikidata',
    author='Pablo Calleja, Alberto Sánchez, Oscar Corcho',
    email='p.calleja@upm.es',
    license='Apache 2',
    python_requrires='>=3.7',
    # packages=find_packages(
    #    where='src',
    #    include=['widaug','widaug.*'],  # alternatively: `exclude=['additional*']`
    # ),
    # package_dir={"": "src"},
    packages=['widaug'],
    package_dir={'widaug': 'src/widaug'},
    # install_requires=[],
    setup_requires=[],
    # install_requires=install_requires,
    include_package_data=True,

    # test_requires=[]
    # test_suite='tests'

)