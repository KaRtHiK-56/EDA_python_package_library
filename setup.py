from setuptools import setup, find_packages

"""with open(r"C:\Users\Devadarsan\Desktop\Karthik_projects\EDA_Package\EDA_python_package_library\README.md", "r") as f:
    long_description = f.read()

setup(
    name="EDA-Python-Library",
    version="0.0.1",
    description="A Library for Making the Explorartory Data Analysis process easy in single line of codes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KaRtHiK-56/EDA_python_package_library",
    author="Karthik",
    author_email="karthiksurya611@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["bson >= 0.5.10"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.10",
)"""

 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3.10.0'
]
 
setup(
  name='EDA-Python-Library',
  version='0.0.2',
  description='A Library for Making the Explorartory Data Analysis process easy in single line of codes',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url="https://github.com/KaRtHiK-56/EDA_python_package_library",  
  author='Karthik',
  author_email='karthiksurya611@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords='eda', 
  packages=find_packages(),
  install_requires=[''] 
)