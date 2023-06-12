from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

if __name__ == "__main__":
    try:
        setup(
            name='smart-crossover',
            version='0.1',
            description='Smart Crossover Algorithms for LP',
            long_description=readme,
            url='https://github.com/wcwj0147/smart-crossover',
            author='Your Name',  # replace with your name
            author_email='wcwj1999@outlook.com',
            packages=find_packages(where='src'),
            package_dir={'': 'src'},
            install_requires=[],
            classifiers=[
                'Programming Language :: Python :: 3.7',
            ],
        )
    except:
        print("\n\nAn error occurred while building the project, "
              "please ensure you have all necessary dependencies installed")
        raise
