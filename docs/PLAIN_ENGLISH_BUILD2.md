# Build 2 in plain English

No maths. This explains what the second build changes and why, for someone who
does not work with these models.

## The situation in one paragraph

We have a working model that forecasts maize prices in fifteen Nigerian markets:
the panel fixed-effects model. Somebody asked whether a neural network could do
better. Two kinds were tried and both lost. The code was then lost, so we are
rebuilding. Build 1 recreates what was tried before. Build 2 asks a fairer
question: if we give these networks every reasonable advantage, do they still
lose? A fair test costs little and it means the answer stands up when somebody
asks whether the networks were given a proper chance.

## What the two networks are

Both read the last 52 weeks of a market's history one week at a time, keeping a
running summary in their head as they go. At the end of the 52 weeks they use that
summary to guess what the price will be in one month, three months and six
months.

The difference between them: the **RNN** has a memory that fades, so the further
back something happened, the fainter it is by the time the model reaches the
present. The **GRU** has two switches that let it decide what to keep and what to
throw away. The textbook says the GRU should win. Last time it lost, and Build 2
does not assume the textbook is right.

## The eleven changes, in order of how much they should matter

### 1. Tell the model to look at last year's price

This is the big one, and it explains why the existing model wins.

The panel model has one column that says "the price in this same week last year."
It cannot ignore that column. When forecasting six months ahead, that year-ago
price is real, recorded history; it is not a guess. So the panel model always has
something solid to hold onto, even far out.

The networks did see last year's price. It was the very first of the 52 weeks they
read. But by the time they had read 51 more weeks and formed their summary, it had
faded to almost nothing, and nothing in their design told them it mattered. They
would have had to work that out for themselves from about six thousand examples,
and they did not.

Build 2 hands it to them directly, as a separate note attached to the summary
rather than something buried at the start of the reading.

There is an extra wrinkle worth understanding. The three forecasts come out of one
shared summary. But "last year, same week" is a different point in the 52 weeks
depending on which forecast you mean:

- For the one-month forecast, the relevant week last year is near the start.
- For the three-month forecast, it is a bit later.
- For the six-month forecast, it is right in the middle.

One summary had to hold three separate anchors at once and each forecast had to
find the right one. Build 2 gives each forecast its own anchor, plainly labelled.

We also add "what did prices do over this same stretch last year," because knowing
where the anchor sits is useful, and knowing which direction it moved from there is
more useful still.

None of this is cheating. Last year's price is known today. Nothing from the future
is used.

**A question that comes up:** doesn't the model already have seasonal information?
Yes, but a different kind. It has four columns describing the *average* shape of a
year: prices usually rise here, usually fall there. Those are identical in 2017 and
2024. Last year's actual price is different information: that year's particular
harvest, that market's particular disruption, the level it was operating at. Two
years can have the same average-shape numbers and last-year prices two hundred naira
apart. We keep both.

### 2. Make the three forecasts equally important

The model is graded on all three forecasts together and tries to reduce its total
mistake.

A six-month forecast is naturally more wrong than a one-month forecast, just
because more can happen in six months. So when the model adds up its mistakes, the
six-month ones dominate the total simply by being bigger, and the model quietly
spends most of its effort there. Nobody decided that; it fell out of the
arithmetic.

Build 2 rescales the three so that "a big mistake" means the same thing at each
horizon, and the model's effort goes where we want rather than where the numbers
happen to push it.

### 3. Stop the model marking its own homework

Training works like this: the model practises on older data, and we hold back
recent data to check whether it is genuinely improving or just memorising. When the
check score stops improving, we stop training. That check is the main thing
preventing the model from memorising, so it needs to be trustworthy.

The problem: each training example includes 52 weeks of history. An example from
just before the held-back period shares almost all of its 52 weeks with the first
held-back examples. So the check is partly made against data the model already
practised on. It looks like an independent exam; it is not.

Build 2 leaves a deliberate gap between the practice material and the exam
material. Wide enough that no example on either side shares any history with the
other.

This costs training data, which is a real loss when there is not much to begin
with. The code reduces the gap if it would leave too little to train on, and
records what it did each time.

### 4. Give the model something to hold it back

The model has roughly four examples per adjustable dial. That is not many. With
that few, a model can fit the past almost perfectly and forecast the future badly,
because it has enough freedom to memorise rather than learn.

Build 1 had one protection: stopping early. That is it. Build 2 adds two standard
ones. The first randomly ignores part of the model's own reasoning during each
practice round, which stops it depending too heavily on any single pattern. The
second gently penalises large dial settings, which keeps it from lunging at
particular examples.

Both are set to conventional values rather than tuned, deliberately. Tuning them
by checking which value scores best on the final exam would be exactly the
cheating the change-control rule exists to prevent.

### 5. Run it seven times instead of twice

These models start from random settings, so the same model trained twice gives
slightly different answers. Last time it was run twice. The prior session recorded
that the difference between two random starts was *larger than the difference
between models being compared*, which means two runs cannot tell you which model
is better.

Build 2 runs seven and takes the middle answer, not the average, so one run that
goes badly wrong cannot drag the result. It also reports the spread, so anyone
reading a result can see how much of it is real.

### 6. Add a sixteenth market to the practice set

The panel has sixteen markets. Only fifteen are scored, because Aba is not in the
existing model's results and we need identical comparisons.

But nothing stops the network *practising* on Aba. The core argument for why these
networks struggle here is that they need many series and we have few, so adding one
is the cheapest possible test of that argument. Aba is still never scored, so it
cannot flatter the result.

Aba has the patchiest record of the sixteen. If it turns out to be more noise than
signal, that will show up.

### 7. Weight recent years more heavily

Prices in this panel run from about 28 naira per kilo to about 1,222, and most of
that climb is 2023 and 2024. Diesel went from about 102 to about 1,931 over the same
period. The recent years are, in a real sense, a different economy.

Treating 2015 and 2024 as equally informative asks one model to describe both.
Build 2 tilts toward recent years without discarding the old ones, which are still
the only evidence we have about seasonal patterns.

### 8. Refit four times a year instead of twice

The model is not rebuilt every week; that would take too long. Between rebuilds it
runs on a slightly out-of-date fit, exactly as it would in real use. Build 1
rebuilt roughly twice a year. Build 2 rebuilds four times, which also matches how
often the project reviews its models.

### 9. Train more patiently

Build 1 took fairly large steps and gave up after fifteen rounds without
improvement. Build 2 takes smaller steps that get smaller still as it goes, and
waits twenty-five rounds before giving up. The aim is to stop because the model has
settled, rather than because it happened to have a bad few rounds.

### 10. Fix the measuring stick

This is not about the models at all, and it was the single biggest mistake last
time.

The existing model's results are recorded for specific market-and-date
combinations. Last time, the challenger generated its own set of dates instead of
using those, and only about a third of them lined up. So the two models were being
graded on different exams, and a headline moved from "14% better" to "25% worse"
without the model changing at all.

Build 1 and Build 2 both read the exam paper from the existing model's own results
file. Nothing is generated. Every table states how many comparisons it rests on.

### 11. Deal with a week the data does not have

One week, 24 April 2019, has no recorded price in any market, yet the existing
model's file reports actual prices for nine forecasts landing there. Those numbers
came from somewhere we cannot see.

Rather than grade against figures we cannot check, we drop them. It costs about half
a percent of the comparisons and it is recorded.

## One thing we deliberately did not do

We did not try lots of model sizes and keep the best. That sounds like leaving
performance on the table, and it might be, but choosing a size by seeing which one
scores best on the final exam means the result is partly a product of that
searching rather than of the model. If a size search is wanted it should be a
separate exercise with its own held-back data.

## A caveat that applies to every number

Both builds are run twice: once where the model must work with only what was known
on the forecast date, and once where it is handed the rainfall, diesel and upstream
prices that actually occurred. The second is always better and it is not achievable
in practice, because nobody knows next March's rainfall.

Awkwardly, the evidence says the existing model's published accuracy was measured
the second way (see `CONDITIONAL_CONVENTION.md`). So the only like-for-like
comparison is the second kind, while the only honest description of real-world
accuracy is the first. Both are reported. Nobody should quote the second kind as an
accuracy anyone will actually get, including for the existing model.

## What to expect

Build 2 should improve most on the six-month forecast, where the missing year-ago
anchor was doing the most damage. It probably still will not clear the bar for
replacing the production model, which requires beating it by more than 5% at both
one month and three months at once, twice in a row.

That is not a wasted exercise. §8.5 of the project handoff predicted this outcome
before any of this work started, on the grounds that these networks need hundreds
of series and we have sixteen. Confirming a prediction that was written down in
advance is worth recording, and it means the next person to propose a neural model
here has a documented answer rather than a hunch.
