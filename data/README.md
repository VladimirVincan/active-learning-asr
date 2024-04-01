How to download all libraries:

```
make
```

How to create sorted dataset in LibreOffice Calc:
select speaker\_id column
Data -> Pivot Table -> Insert or edit
Remove Data from Column fields.
Drag speaker\_id to Row Fields and Data Fields.
Press OK
In sheet with pivot table, create column count (third column) that is the division of sum-speaker\_id column with speaker\_id column.
Then, in original sheet, create count column with the following formula: VLOOKUP(A2,Sheet2.$A$2:$B$1000,3,0)
