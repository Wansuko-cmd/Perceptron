SOURCE=$(cat src/commonMain/kotlin/com/wsr/scope/IOScope.kt)
RESULT=""
while read line; do
  RESULT+="${line//TEMP/$RANDOM}"
  RESULT+='
  '
done <<< "$SOURCE"
echo "$RESULT" > src/commonMain/kotlin/com/wsr/scope/IOScope.kt
