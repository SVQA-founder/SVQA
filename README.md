# SVQA
The SVQA(Synthetic Video Question Answering) dataset contains 12000 videos and around 120k QA pairs. Videos and QA pairs are all generated automatically with minimal language biases and clearly defined question categories. The dataset can facilitate the analysis on models reasoning skills.

# Video and QA Pair Examples
| QA Category|Question|Answer|Video(GIF)
| :----------------- | --------------------------------- | ----------------------------- | ---------------------------------------- |
|Attribute Comparison|![](GIF/6.jpg)|no|![](GIF/3997.gif)|
|Count|How many cylinders behind the big cylinder that moves left?|5|![](GIF/377.gif)|
|Query|What is the color of the object that is rotating behind the black object, and to the right of the yellow object in the beginning?|blue|![](GIF/1792.gif)|
|Integer Comparison|Are there more balls that are moving backward than small red cylinders?|no|![](GIF/4929.gif)|
|Exist|A green cylinder shows up, does a gray cylinder that rotates whose starting time earlier than it?|yes|![](GIF/6517.gif)|


## Statistics of SVQA
| Question Category       |Sub Category|       Train |       Val  |       Test  |
| :---------------        |:------     | ----------: | ---------: | ----------: |
| **Count**               |            |     19320   |      2760  |  5520       |
| **Exist**               |            |     6720    |      960   |  1920       |
|**Query**                |Color       |     7560    |      1056  |  2160       | 
|                         |Size        |     7560    |      1056  |  2160       | 
|                         |Action Type |     6720    |      936   |  1920       | 
|                         |Direction   |     7560    |      1056  |  2160       | 
|                         |Shape       |     7560    |      1056  |  2160       |
|**Integer Comparison**   |More        |     2520    |      600   |  720        | 
|                         |Equal       |     2520    |      600   |  720        | 
|                         |Less        |     2520    |      600   |  720        |
|**Attribute Comparison** |Color       |     2520    |      216   |  720        | 
|                         |Size        |     2520    |      216   |  720        | 
|                         |Action Type |     2520    |      216   |  720        | 
|                         |Direction   |     2520    |      216   |  720        | 
|                         |Shape       |     2520    |      216   |  720        | 
| **Total QA pairs**      |            |     83160   |      11880 |  23760      |
| **Total Videos**        |            |     8400    |      1200  |  2400       |
