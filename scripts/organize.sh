cd dataset

cd LOSS/

ls | sort -R | tail -185 | while read file; do
  mv $file ../test_set/loss_edits/
done

mv * ../training_set/loss_edits/

cd ../reducciones

ls | sort -R | tail -248 | while read file; do
  mv $file ../test_set/not_loss/
done

mv * ../training_set/not_loss/

cd ../..
