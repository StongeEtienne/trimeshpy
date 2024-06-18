import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIREMENTS = ["numpy>=1.24.0",
                "scipy>=1.10.0",
                "h5py>=3.7.0",
                "vtk>=9.2.0",
                "tqdm>=4.29.0",
                "nibabel>=5.2.0"],

setuptools.setup(
    name='trimeshpy',
    packages=setuptools.find_packages(),
    install_requires=REQUIREMENTS,
    version='0.0.3',
    license='MIT',
    author="Etienne St-Onge",
    author_email="Firstname.Lastname@usherbrooke.ca",
    description="Triangular Mesh Processing in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StongeEtienne/trimeshpy",
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
)
