# chronos

Can a circumbinary planet reveal itself through eclipse-timing variations, even when TESS observes only in separated windows? This exploratory script generates synthetic eclipsing-binary properties, injects planets, samples eclipse times that fall within TESS observing windows, adds a simple timing-noise estimate, and searches the resulting signal with a Lomb-Scargle periodogram. Its target and planet distributions are modeling assumptions, so treat its output as a simulation study rather than an observed recovery rate. The script reads `TESS_orbit_times.csv`, which is not included in this repository; it does not load source, lens, or Roman noise tables.

Set `CHRONOS_PATH` to the checkout directory when running the sampler from elsewhere. If unset, Chronos uses its checkout root. The current script reads `TESS_orbit_times.csv` from the ignored `data/` directory and writes `etv_sample.csv` there. Generated figures can go in the ignored `visuals/` directory.
