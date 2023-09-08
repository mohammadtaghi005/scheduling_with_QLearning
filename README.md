# scheduling_with_QLearning

This is a discrete event simulation of a healthcare facility, implemented in Python.


## Overview

The simulation models patients arriving to the facility and going through different queues and services like checking in, seeing a general doctor, seeing a specialist, getting lab work done, etc. It tracks queue lengths, waiting times, server utilizations, and other key performance metrics.

The main simulation logic is in simmulation_healthcare_v9(important_changes).py. Key components include:

1. Helthcare_Simulation class - Defines the core simulation model and logic
2. Event functions like refer_start(), refer_end(), etc - Handle different event types
3. data_def() - Initializes data structures to track simulation state
4. fel_maker() - Adds new future events to the event list
5. calculate_kpi() - Calculates key performance indicators after simulation runs
6. warm_up() - Runs multiple replications to find warm-up period
## Customizing the Simulation

The main parameters to modify are:

- Number of servers
- Arrival distribution
- Service time distribution
- Reservation probabilities
- Penalty weights
These are configured in the Helthcare_Simulation initialization and in data_def().

Additional enhancements could include:

- More complex patient pathways
- Resource scheduling algorithms
- Animation of simulation
- Larger data sets and analysis

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Please refer to the [LICENSE](LICENSE) file for more details.
