# Google Reviews Scraper

Tool to extract google reviews for ai fine-tuning and research purposes.


This will get all reviews that are visible on the page.

### steps:

1. **open the google reviews page**
   - go to: https://share.google/rf2Ec6jS5V4IMRRKK
   - or search for "Crimson Coward Fredericksburg" on google maps
   - click on the business
   - scroll down to the reviews section

2. **load all reviews**
   - press down arrow to scroll down and load more reviews
   - scroll down repeatedly until no more reviews load
   - this may take a while if there are many reviews

3. **extract reviews**
   - open browser console (f12 or right-click > inspect > console)
   - copy the code from `simple_extract.js`
   - paste into console and press enter
   - wait for it to finish

4. **save the output**
   - copy the json output from the console
   - save it as `reviews.json`


The json file contains all the reviews that were visible on the page.




