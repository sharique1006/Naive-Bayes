question=$1
train_data=$2
test_data=$3
output_file=$4

if [ ${question} == "1" ] ; then
	python3 Q1/nbbest.py $train_data $test_data $output_file
elif [ ${question} == "2" ] ; then
	python3 Q2/svmbest.py $train_data $test_data $output_file
elif [ ${question} == "3" ] ; then
	python3 Q3/bestsgd.py $train_data $test_data $output_file
fi