# r2d2 Directory

This directory contains two important files:


## foldertoimages.sh

This is a shell script that processes APK files. It measures the size of each APK file, executes a Python script (`apktoimage.py`) on it, and logs the execution time and APK size. It also logs the names of the APK files that have been processed.

### Usage

```bash
./foldertoimages.sh [APK paths file] [Output folder] [Log file]
```

2. `apktoimage.py`: This is a Python script that is called by `foldertoimages.sh`. It takes an APK file and an output folder as arguments. The exact functionality depends on the implementation of this script.

   Usage: `python3 apktoimage.py [APK file] [Output folder]`

Please ensure that you have the necessary permissions to execute these scripts and that the required dependencies are installed.