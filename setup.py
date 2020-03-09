import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIREMENNTS = ["numpy>=1.16.2",
                 "scipy>=1.0",
                 "h5py>=2.5.0",
                 "vtk>=8.1.2",
                 "tqdm>=4.29.0",
                 "nibabel>=2.4.0"],

setuptools.setup(
    name='trimeshpy',
    packages=setuptools.find_packages(),
    install_requires=REQUIREMENNTS,
    version='0.0.1b1',
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
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
