# Berkeley Innovation Index
#### Automatic Report Generator



Code (Python and HTML) for running a Flask web application that automatically creates visual reports for BII test takers.

The two main applications are:
- fetch.py collects data in realtime from a Google Spreadsheet and stores it at the server as a JSON datafile.
- main.py renders templates with graphical reports of the result for Individuals and/or Workgroups.
