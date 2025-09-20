echo "how many do you need sir"
read x

if [ $x -gt 10 ]; then
	echo "bhai, itni files kyoun banwaani hai? kya chul hai"
else
	for ((i = 1; i < $x; i++)); do
		touch awesomekriss$i;
	done
fi


