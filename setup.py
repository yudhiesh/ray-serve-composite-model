import setuptools

setuptools.setup(
    name="heart-failure",  # Replace with your own username
    version="0.0.1",
    author="Manning",
    description="News Category",
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'pygit2==1.6.0',
        'aiohttp==3.7.4',
        #if using python 3.6 uncomment dataclasses
        #'dataclasses==0.7',
        'numpy==1.19.2',
        'unidecode==1.2.0',
        'ray[default]==1.8.0',
        'ray[serve]==1.8.0',
        'pyre-check==0.9.3',
        'pydantic>=1.8',
        'python-Levenshtein',
        'numba',
        'scikit-learn==1.0.2',
        'future_annotations',
        'cloudpickle',
        'starlette',
        'fastapi',
        'pandas',
        'python-multipart',
        'transformers==4.10',
        'nltk',
        'tensorflow==2.5'
    ],
)
