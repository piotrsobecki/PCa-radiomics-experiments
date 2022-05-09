from setuptools import setup, find_packages

with open('LICENSE') as f:
    license = f.read()

setup(name='prostatex',
      version='0.3',
      description='ProstateX',
      url='https://bitbucket.org/piotrsobecki/prostatex',
      author='Piotr Sobecki',
      author_email='ptrsbck@gmail.com',
      license=license,
      packages=find_packages(exclude=('tests','scripts', 'docs')),
      install_requires=['mahotas','pydicom','opt==0.4'],
      dependency_links=['http://github.com/piotrsobecki/opt/tarball/master#egg=opt-0.4'],
      zip_safe=False)