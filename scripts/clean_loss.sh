declare -a arr=($(find . -type d | grep "./" | tr '\n' ' '))

for i in "${arr[@]}"
do
  cd $i
  # Remove 0 bytes length files
  rm $(find . -size 0 | tr '\n' ' ')
  # Rename pngs and remove non png/jpg files
  python ../clean_files.py
  cd ..
done
