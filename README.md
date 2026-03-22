# ARC (Anything Retrieved for Community)
As Google spends all of its time and everyone else's resources on AI, more specifically LLMs, our ability to see through the corporate internet is waning.
Here, we will build a new standard to take the internet back. We should strive to create a distributed network of devices hosting this engine and to provide exactly what anyone is looking for. No "sponsored results" or AI-summaries, just the web.  

## Goals
1. Hosting  
Currently, www.arcsearch.net is the only webpage which serves search results and the webpage. I'd like to allow this to simply be an API which can pull results from many system shards and allow others to use their own domains to host their own interfaces of the same system.
2. Distros  
As it sits, the system is intended for Debian systems. I'd like any Linux distro or even Windows/MacOS to be able to contribute.
3. Primary Servers  
Currently, only my server handles routing and acts as a central connection for all of the worker systems. I'd like to be able to have more than one central system in this in case of outages.  
4. Obtain Trusted Reviewers  
I need people who can eventually take some of the workload from me as this grows. We need trusted code and eventually I won't be able to audit as well as I'd like.

## Standards
1. The worker installer should be kept stand-alone. It should be a one-time run which checks all system resources.
2. Shards cannot have unique data. Every system should host copies in order to ensure no results will be lost in outages.
3. The installs need regular validation from other trusted systems. We can't have anyone tampering with data.
