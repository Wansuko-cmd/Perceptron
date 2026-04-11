TARGET=$(find . -type f | grep 'src/commonMain/kotlin/com/wsr' | grep -v 'scope')
SOURCE=""
for FILE in $TARGET; do
  SOURCE+=$(cat $FILE | grep -v 'package' | grep -v 'import')
  SOURCE+='
  '
done
echo "$SOURCE"
