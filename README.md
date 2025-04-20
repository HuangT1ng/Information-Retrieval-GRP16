## To developer

Download solr and set up with your jdk-11 to run the solr.

jdk-11:https://www.oracle.com/java/technologies/downloads/#java11?er=221886

solr:https://solr.apache.org/

### Run solr
To run solr in localhost, download the Binary releases.

Use the command line in the ./solr/bin


```bash
solr.cmd start
```

If it is the first time to run the solr, use following command line to create the core

```bash
solr create -c opinion_search
```

Then, run the index_data.py.

### Run the query web

Run the app.py.