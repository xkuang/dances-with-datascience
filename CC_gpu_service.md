# Compute Canada GPU resources

## Rapid Access Service
[Any Compute Canada user can access modest quantities of compute, storage and cloud resources as soon as they have a Compute Canada account.](https://www.computecanada.ca/research-portal/accessing-resources/rapid-access-service/) 

Any user can log in and use a shared pool of resources set aside for Rapid Access Service users. This provides users immediate access, even if their work was not included in a Resource Allocation Call (that is, a formal request for resources, which takes place on an annual basis at Compute Canada). Rapid Access Service users have access to about 10% of the total compute pool, both for CPU and GPU resources. The resource limits are 50 CPU years and 10 GPU years per user. 

## Getting an account

In order to make use of the Rapid Access Service, users must have a CCDB account. Users register through this [link](https://www.westgrid.ca/support/accounts/registering_ccdb) and require a sponsor. A sponsor is typically a principal investigator who is already registered with Compute Canada. If the sponsor is in place, account creation can be very fast as a user simply applies and as soon as the sponsor approves the request, the account is created. 

If a sponser is not set up yet, the sponsor needs to register first. 

## Getting started

Compute Canada makes its documentation available through [this wiki.](https://docs.computecanada.ca) While there are general purpose cloud resources available, all of Compute Canada's GPUs are currently only available via their batch system. As such, users must log into one of the compute resources via SSH in order to submit a GPU enabled job. 

For example, information on accessing the Cedar system at SFU can be found [here](https://docs.computecanada.ca/wiki/Cedar). 

## Submitting a tensorflow job

Compute Canada also provides documentation on running [tensorflow jobs](https://docs.computecanada.ca/wiki/Tensorflow) using the SLURM job management system. The documentation instructs users on how to set up tensorflow in a virtual environment. Once set up, the user can easily submit their tensorflow python script by simply copying it and submitting it via the job submission script. 

## Quota and scheduling 

Unfortunately, at the time of testing, the GPUs on the Cedar (and Graham) cluster was in full use and the sample tensorflow script did not run immediately. This is in part because users not utilizing the Rapid Access Service, but with dedicated allocations have priority. At the time of submission, there were over 360 other jobs ahead in the queue. After approximately 3 days, the job was submitted and ran. 

Scheduling is based on a "fair share" policy. Because all Rapid Access Service users share the same allocation and the implementation of the "fair share" policy, a user's ability to use the service is affected by how active other Rapid Access Service users have been. A neutral fair share score is 0.50. That is, the account has used its exact target allocation. Lower scores mean the user has used too much allocation. Currently, the Rapid Access Service's fair share score is 0.03, effectively meaning any Rapid Access Service users' jobs are very low in priority.  

## Conclusion

Gaining access and logging into Compute Canada's Rapid Access Service is straight forward and the documentation is well set up to get users up and running. The system is being used heavily and given the backlog of jobs in the queue and the low priority assigned to Rapid Access Service jobs, it service is likely better suited for production level jobs, rather than for iterative tuning of deep learning training algorithms. While it is unlikely Rapid Access Cloud GPU users would be able to do their development work on the Rapid Access Service, it is conceivable that researchers lean on the Rapid Access Service for their longer term training jobs in order to free up resources on the Rapid Access Cloud. 
