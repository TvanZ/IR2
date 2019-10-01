(1)  Execute click model for randomization model (Make separate script)

    Instructions from Sietze:
            (1) rename swapped weights so .weights (one at a time b/c otherwise bugs)
                          --->  can only have one .weights file in a folder at a time

             in main.py of GANoN

(2) Create a sub_dir of randomization_results

    Contains query_ID1.txt, query_ID2.txt, ....

    For each query_ID.txt: have


        file ID1    trial1_position, clicked?        ...        trial100_position, clicked?
          .
          .
          .

        file ID10   trial1_position, clicked?        ...        trial100_position, clicked?


 (3) Have some script that process the data in these sub-directories to get

    (a) Relevance score per file
    (b) Visualize the results?