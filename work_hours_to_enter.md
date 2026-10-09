# Work hours to enter — purop.birzeit.edu/student/projects/486/work-hours

16 entries, 56.5 hours total, 25 Jul to 15 Aug 2026.
Copy the description text directly. Check the two date fields (start / end) match
the times shown.

NOTE: the log already contains a duplicate (11 Jul submitted twice, two seconds
apart). Submit once and wait for the row to appear before starting the next.

---

## 1. Sat 25 Jul 2026 — 10:00 to 14:00 (4.0 hrs)

trying to get straight what counts as quantum and what doesn't. read QAOA, then quantum annealing, then found simulated bifurcation. took me way too long to realise SB isn't quantum at all, it runs on a normal CPU. the Goto paper says classical mechanics right in the title.

## 2. Mon 27 Jul 2026 — 19:30 to 21:30 (2.0 hrs)

checked if real quantum hardware was an option. D-Wave free tier is about a minute of QPU and caps around 177 fully connected variables. ours is far bigger. so quantum inspired it is.

## 3. Wed 29 Jul 2026 — 20:00 to 22:00 (2.0 hrs)

data hunting. FAOSTAT has Palestine prices and yields back to the 90s. tried ARIJ too but couldn't get anything tabular out of the reports.

## 4. Sat 1 Aug 2026 — 09:30 to 14:30 (5.0 hrs)

PCBS. found crop production by governorate and irrigation type, plus the 2023 price survey. Jenin has the most cultivated area of any governorate. saved all the URLs.

## 5. Mon 3 Aug 2026 — 19:30 to 21:30 (2.0 hrs)

downloaded the Kaggle crop dataset, everyone cites it. it's Indian and only 3 of our 10 crops are in it. no idea yet what to do about wheat, tomatoes, olives.

## 6. Wed 5 Aug 2026 — 20:00 to 22:00 (2.0 hrs)

more SB benchmark papers. SA is the standard baseline. started sketching the QUBO.

## 7. Thu 6 Aug 2026 — 20:30 to 22:00 (1.5 hrs)

short one. OCHA has the A/B/C boundaries on HDX. downloaded. haven't opened it.

## 8. Fri 7 Aug 2026 — 09:00 to 13:00 (4.0 hrs)

pulled the PCBS tables. Jenin rainfed 2.72 t/ha vs irrigated 18.4. didn't expect the gap to be that big.

## 9. Sat 8 Aug 2026 — 10:00 to 14:30 (4.5 hrs)

OSM extract from Geofabrik, pulled farmland and orchard polygons for Jenin. 941.

## 10. Sun 9 Aug 2026 — 19:30 to 22:00 (2.5 hrs)

NASA POWER fetch, 2015 to 2024. rainfall came out 15 mm a year. obviously broken. PRECTOTCORR is mm per day not per month. fixed, 465 mm.

## 11. Mon 10 Aug 2026 — 19:00 to 21:30 (2.5 hrs)

first QUBO with slack variables for the caps. wrote a brute force checker on 5x3 so I can see the true optimum.

## 12. Tue 11 Aug 2026 — 19:30 to 22:30 (3.0 hrs)

88% and I can't get past it. the slack can't separate breaking the cap by a bit from just not representing the leftover. turn the penalty up and it breaks the other way.

## 13. Wed 12 Aug 2026 — 19:00 to 22:00 (3.0 hrs)

rewrote it. demand is a price effect now, water is charged per m3. one penalty left instead of three. checker at 100%.

## 14. Thu 13 Aug 2026 — 20:30 to 22:00 (1.5 hrs)

clipped the parcels to the Oslo zones. 625 of 941 were inside Israel. 174 left in A/B.

## 15. Fri 14 Aug 2026 — 09:00 to 17:00 (8.0 hrs)

dSB and SA written. dSB is a mess at full size, 136 of 174 parcels ending up with more than one crop. found the domain wall paper late on, tried it, violations went to zero.

## 16. Sat 15 Aug 2026 — 09:00 to 18:00 (9.0 hrs)

SoilGrids per parcel, RF trained. LSTM loses to just using last year's price so it goes in as a negative result. 2024 FAOSTAT veg prices are 3 to 5x normal, war disruption, dropping the year.
