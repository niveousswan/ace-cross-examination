"""ACE benchmark: teacher-framed cases with admissible evidence (F3), an AI
"witness" answer whose claims are labelled sound / flawed, and a re-asked
answer used by the consistency probe (Q5).

Every flawed claim carries the error type a teacher would record when seeding
or spotting it. Cases are split by case (never by claim) into a development
set, used only to tune thresholds, and a held-out test set.

Flaw types
----------
number          wrong quantity, date or count
substitution    wrong entity / category member (person, place, organelle ...)
direction       reversed relation (higher/lower, absorb/reflect ...)
negation        polarity reversed (can / cannot)
overgeneral     unlicensed absolute (always, never, only, entire ...)
fabricated_cit  citation to a source that is not admissible / does not exist
undefined_form  formula with undefined symbols (not testable as stated)
vague           hedged, untestable claim ("many historians believe ...")
unsupported     invented detail with no support in the evidence
misconception   common misconception not reducible to one of the above
relational      wrong relation between entities that all appear in evidence
"""

S, F = "sound", "flawed"


def c(text, label=S, flaw=None):
    return {"text": text, "label": label, "flaw": flaw}


CASES = [
    # ------------------------------------------------------------------ DEV
    {
        "id": "S1", "split": "dev", "subject": "science",
        "question": "Why do a bowling ball and a feather fall differently on Earth but the same way on the Moon?",
        "sources": [
            {"id": "openstax-phys-2.7", "author": "OpenStax", "title": "College Physics, Sec. 2.7 Falling Objects", "year": 2012, "text": [
                "In the absence of air resistance, all objects fall with the same constant acceleration, regardless of their mass.",
                "Near Earth's surface the acceleration due to gravity is about 9.8 m/s².",
                "Air resistance opposes the motion of a falling object and depends on its speed, size and shape.",
                "A feather falls more slowly than a bowling ball on Earth because air resistance is large relative to the feather's weight."]},
            {"id": "nasa-apollo15", "author": "NASA", "title": "Apollo 15 Mission Record", "year": 1971, "text": [
                "During the Apollo 15 mission in 1971, astronaut David Scott dropped a hammer and a feather on the Moon and they hit the surface at the same time.",
                "The Moon has no significant atmosphere, so there is no air resistance.",
                "The acceleration due to gravity on the Moon is about 1.6 m/s², roughly one sixth of Earth's."]},
        ],
        "answer": [
            c("On Earth, the feather falls more slowly because air resistance is large compared with its weight."),
            c("The drag force is given by F = kv^2, which slows the feather.", F, "undefined_form"),
            c("The Moon has no significant atmosphere, so there is no air resistance there."),
            c("On the Moon, both objects accelerate at about 3.7 m/s².", F, "number"),
            c("A 2003 study confirmed that the hammer and feather landed together (Harris, 2003).", F, "fabricated_cit"),
            c("Without air resistance, all objects fall with the same acceleration regardless of mass."),
        ],
        "reask": [
            "Air resistance has a much bigger effect on a feather than on a bowling ball on Earth.",
            "Drag depends on speed, size and shape.",
            "There is essentially no atmosphere on the Moon.",
            "Lunar gravity gives both objects an acceleration of about 1.6 m/s².",
            "Astronaut David Scott showed this during Apollo 15.",
            "In a vacuum, mass does not change how fast an object falls.",
        ],
    },
    {
        "id": "S2", "split": "dev", "subject": "science",
        "question": "What happens during photosynthesis?",
        "sources": [
            {"id": "openstax-bio-8", "author": "OpenStax", "title": "Biology 2e, Ch. 8 Photosynthesis", "year": 2018, "text": [
                "Photosynthesis takes place in the chloroplasts of plant cells.",
                "During photosynthesis, plants use light energy to convert carbon dioxide and water into glucose and oxygen.",
                "Chlorophyll absorbs mainly red and blue light and reflects green light, which is why leaves look green.",
                "The light-dependent reactions occur in the thylakoid membranes, and the Calvin cycle occurs in the stroma."]},
            {"id": "campbell-bio", "author": "Urry et al.", "title": "Campbell Biology", "year": 2020, "text": [
                "Plants also carry out cellular respiration, which releases carbon dioxide, in their mitochondria.",
                "Oxygen released during photosynthesis comes from splitting water molecules."]},
        ],
        "answer": [
            c("Photosynthesis happens in the chloroplasts."),
            c("Plants take in oxygen and release carbon dioxide during photosynthesis.", F, "misconception"),
            c("Chlorophyll absorbs mainly green light, which is why leaves look green.", F, "relational"),
            c("The oxygen given off comes from water molecules being split."),
            c("Plants never release carbon dioxide.", F, "overgeneral"),
        ],
        "reask": [
            "Chloroplasts are where photosynthesis takes place.",
            "In photosynthesis, plants absorb carbon dioxide and give off oxygen.",
            "Leaves look green because chlorophyll reflects green light.",
            "Splitting water releases the oxygen.",
            "Plants do not give off carbon dioxide.",
        ],
    },
    {
        "id": "S3", "split": "dev", "subject": "science",
        "question": "What causes the seasons?",
        "sources": [
            {"id": "nasa-seasons", "author": "NASA Space Place", "title": "What Causes the Seasons?", "year": 2021, "text": [
                "Earth's seasons are caused by the tilt of its rotational axis, which is about 23.5 degrees.",
                "When the Northern Hemisphere is tilted toward the Sun, it has summer and the Southern Hemisphere has winter.",
                "Earth is actually closest to the Sun in early January, during the Northern Hemisphere's winter.",
                "Seasons are not caused by changes in Earth's distance from the Sun."]},
            {"id": "openstax-astro-4", "author": "OpenStax", "title": "Astronomy, Ch. 4", "year": 2016, "text": [
                "Sunlight strikes the ground more directly in summer, so each square metre receives more energy.",
                "Days are longer in summer than in winter."]},
        ],
        "answer": [
            c("The seasons are caused by the tilt of Earth's axis."),
            c("Summer happens when Earth is closest to the Sun.", F, "misconception"),
            c("Earth's axis is tilted by about 45 degrees.", F, "number"),
            c("When it is summer in the Northern Hemisphere, it is winter in the Southern Hemisphere."),
            c("In summer the Sun's rays hit the ground at a steeper angle, delivering more energy per square metre."),
        ],
        "reask": [
            "Axial tilt is what produces the seasons.",
            "Earth is nearest the Sun in January.",
            "The tilt of Earth's axis is about 23.5 degrees.",
            "The two hemispheres have opposite seasons.",
            "Summer sunlight arrives more directly.",
        ],
    },
    {
        "id": "S4", "split": "dev", "subject": "science",
        "question": "Why does water boil at a different temperature on a mountain?",
        "sources": [
            {"id": "openstax-chem-10.3", "author": "OpenStax", "title": "Chemistry 2e, Sec. 10.3 Phase Transitions", "year": 2019, "text": [
                "A liquid boils when its vapour pressure equals the surrounding atmospheric pressure.",
                "At sea level, pure water boils at 100 °C.",
                "Atmospheric pressure decreases as altitude increases, so water boils at a lower temperature at high altitude.",
                "At the top of Mount Everest, water boils at about 70 °C."]},
            {"id": "usda-altitude", "author": "USDA", "title": "High Altitude Cooking and Food Safety", "year": 2020, "text": [
                "Because water boils at a lower temperature at high altitude, foods cooked in boiling water take longer to cook."]},
        ],
        "answer": [
            c("Water boils when its vapour pressure equals the atmospheric pressure."),
            c("At high altitude, water boils at a higher temperature because the air is thinner.", F, "direction"),
            c("At sea level, pure water boils at 100 °C."),
            c("On the summit of Everest, water boils at roughly 50 °C.", F, "number"),
            c("Adding salt is the main reason pasta takes longer to cook in the mountains.", F, "unsupported"),
            c("Food takes longer to cook in boiling water at high altitude."),
        ],
        "reask": [
            "Boiling occurs once vapour pressure matches the pressure of the air above.",
            "Thinner air at altitude means water boils at a lower temperature.",
            "Pure water boils at 100 °C at sea level.",
            "On Everest, the boiling point is about 50 °C.",
            "Cooking in boiling water is slower at high altitude.",
        ],
    },
    {
        "id": "S5", "split": "dev", "subject": "science",
        "question": "What is the structure of DNA and who discovered it?",
        "sources": [
            {"id": "openstax-bio-14", "author": "OpenStax", "title": "Biology 2e, Ch. 14 DNA Structure", "year": 2018, "text": [
                "DNA is a double helix made of two strands of nucleotides.",
                "In DNA, adenine pairs with thymine and guanine pairs with cytosine.",
                "The double-helix structure of DNA was described by James Watson and Francis Crick in 1953, using X-ray diffraction images produced by Rosalind Franklin.",
                "RNA contains uracil instead of thymine."]},
        ],
        "answer": [
            c("DNA has a double-helix structure made of two strands."),
            c("In DNA, adenine pairs with uracil.", F, "substitution"),
            c("Watson and Crick described the double helix in 1962.", F, "number"),
            c("Watson and Crick relied on X-ray images made by Rosalind Franklin."),
            c("Guanine pairs with cytosine."),
        ],
        "reask": [
            "DNA is shaped like a twisted ladder of two strands.",
            "Adenine always pairs with thymine in DNA.",
            "The structure was published in 1962.",
            "Franklin's X-ray diffraction work was key evidence.",
            "Cytosine pairs with guanine.",
        ],
    },
    {
        "id": "S6", "split": "dev", "subject": "science",
        "question": "How does sound travel through different materials?",
        "sources": [
            {"id": "openstax-phys-17.2", "author": "OpenStax", "title": "College Physics, Sec. 17.2 Speed of Sound", "year": 2012, "text": [
                "Sound is a mechanical wave and needs a medium to travel through.",
                "Sound cannot travel through a vacuum.",
                "In air at 20 °C, sound travels at about 343 m/s.",
                "Sound generally travels faster in liquids than in gases, and faster still in solids.",
                "In water, sound travels at about 1480 m/s."]},
        ],
        "answer": [
            c("Sound needs a medium such as air or water to travel."),
            c("Sound can travel through the vacuum of space.", F, "negation"),
            c("Sound moves at roughly 343 m/s in air at room temperature."),
            c("Sound travels more slowly in solids than in gases.", F, "direction"),
            c("According to the Hartmann Acoustic Survey (2015), sound travels at 1480 m/s in water.", F, "fabricated_cit"),
        ],
        "reask": [
            "A medium is required for sound waves.",
            "Space is silent because there is no medium for sound.",
            "In air, sound covers about 343 metres each second.",
            "Solids carry sound faster than gases.",
            "Sound travels at about 1480 m/s in water.",
        ],
    },
    {
        "id": "H1", "split": "dev", "subject": "humanities",
        "question": "Who invented the printing press and why did it matter?",
        "sources": [
            {"id": "eisenstein", "author": "Eisenstein", "title": "The Printing Revolution in Early Modern Europe", "year": 1983, "text": [
                "Johannes Gutenberg developed a movable-type printing press in Mainz around 1440.",
                "The Gutenberg Bible was completed around 1455.",
                "Printing made books cheaper and helped spread the ideas of the Reformation.",
                "Movable type had been used earlier in China and Korea, where Bi Sheng invented movable type around 1040."]},
        ],
        "answer": [
            c("Gutenberg developed his movable-type press in Mainz around 1440."),
            c("Gutenberg built his first press in Paris.", F, "substitution"),
            c("Gutenberg was the first person ever to use movable type.", F, "overgeneral"),
            c("The Gutenberg Bible was finished around 1455."),
            c("Cheaper printed books helped the ideas of the Reformation spread."),
        ],
        "reask": [
            "Around 1440, in Mainz, Gutenberg created a press with movable type.",
            "His workshop was in Mainz.",
            "Movable type existed in China before Gutenberg.",
            "The Gutenberg Bible dates from about 1455.",
            "Printing helped spread Reformation ideas.",
        ],
    },
    {
        "id": "H2", "split": "dev", "subject": "humanities",
        "question": "When and why did the French Revolution begin?",
        "sources": [
            {"id": "openstax-wh2-7", "author": "OpenStax", "title": "World History Volume 2, Ch. 7", "year": 2022, "text": [
                "The French Revolution began in 1789.",
                "On 14 July 1789, crowds in Paris stormed the Bastille.",
                "King Louis XVI was executed in January 1793.",
                "The Declaration of the Rights of Man and of the Citizen was adopted in August 1789.",
                "France faced a severe financial crisis, partly caused by debts from wars, including its support for the American Revolution."]},
        ],
        "answer": [
            c("The French Revolution began in 1789."),
            c("The Bastille was stormed on 14 July 1791.", F, "number"),
            c("King Louis XIV was executed in 1793.", F, "substitution"),
            c("Many historians believe the revolution was mostly about bread prices.", F, "vague"),
            c("War debts, including from backing the American Revolution, contributed to France's financial crisis."),
        ],
        "reask": [
            "It started in 1789.",
            "The storming of the Bastille took place in July 1789.",
            "Louis XVI was executed in 1793.",
            "Bread prices were one of several causes.",
            "France's debts from wars, such as aiding the American colonists, worsened its finances.",
        ],
    },
    {
        "id": "H3", "split": "dev", "subject": "humanities",
        "question": "Can you see the Great Wall of China from space?",
        "sources": [
            {"id": "nasa-wall", "author": "NASA", "title": "China's Wall Less Great in View from Space", "year": 2004, "text": [
                "Yang Liwei, China's first astronaut, reported in 2003 that he could not see the Great Wall from orbit.",
                "The Great Wall is not visible to the naked eye from the Moon.",
                "From low Earth orbit, the wall is very hard to see because it is narrow and similar in colour to its surroundings."]},
            {"id": "britannica-wall", "author": "Encyclopaedia Britannica", "title": "Great Wall of China", "year": 2024, "text": [
                "The Great Wall was built over many centuries by several Chinese dynasties.",
                "Most of the existing wall was built during the Ming dynasty."]},
        ],
        "answer": [
            c("The Great Wall is the only human-made structure visible from the Moon.", F, "misconception"),
            c("Yang Liwei said he could not see the wall from orbit in 2003."),
            c("Most of the wall that survives today was built during the Qin dynasty.", F, "substitution"),
            c("The wall is narrow and blends in with its surroundings, which makes it hard to see from orbit."),
            c("Several dynasties built the wall over many centuries."),
        ],
        "reask": [
            "Astronauts have said the wall cannot be seen from the Moon.",
            "China's first astronaut reported he could not see it.",
            "The Ming dynasty built most of the surviving wall.",
            "It is hard to spot because it is thin and the same colour as the land around it.",
            "Construction spanned many dynasties.",
        ],
    },
    {
        "id": "H4", "split": "dev", "subject": "humanities",
        "question": "What was Magna Carta?",
        "sources": [
            {"id": "bl-magna", "author": "British Library", "title": "Magna Carta: an introduction", "year": 2015, "text": [
                "Magna Carta was sealed by King John at Runnymede in June 1215.",
                "It was a peace treaty between the king and a group of rebel barons.",
                "Pope Innocent III annulled the charter only a few weeks later.",
                "Clause 39 said that no free man could be imprisoned except by the lawful judgement of his peers or the law of the land.",
                "Magna Carta was reissued several times, including in 1216, 1217 and 1225."]},
        ],
        "answer": [
            c("King John sealed Magna Carta at Runnymede in 1215."),
            c("Magna Carta gave every person in England the right to vote.", F, "overgeneral"),
            c("The Pope annulled it within weeks."),
            c("It was never reissued after 1215.", F, "negation"),
            c("Magna Carta was agreed between King Henry and the barons.", F, "substitution"),
        ],
        "reask": [
            "In June 1215 at Runnymede, King John put his seal on Magna Carta.",
            "It protected free men from unlawful imprisonment.",
            "Pope Innocent III cancelled it a few weeks later.",
            "It was reissued in 1216, 1217 and 1225.",
            "The charter was a peace deal between King John and rebel barons.",
        ],
    },
    {
        "id": "H5", "split": "dev", "subject": "humanities",
        "question": "When did the Roman Empire fall?",
        "sources": [
            {"id": "openstax-wh1", "author": "OpenStax", "title": "World History Volume 1, Ch. 15", "year": 2022, "text": [
                "The Western Roman Empire is traditionally said to have ended in 476 CE, when Odoacer deposed the emperor Romulus Augustulus.",
                "The Eastern Roman Empire, often called the Byzantine Empire, lasted until 1453, when Constantinople fell to the Ottomans.",
                "Historians point to many causes for the decline of the Western Empire, including political instability, economic problems and pressure from migrating peoples."]},
        ],
        "answer": [
            c("The Western Roman Empire is traditionally dated as ending in 476 CE."),
            c("The entire Roman Empire collapsed in 476.", F, "overgeneral"),
            c("Romulus Augustulus was deposed by Attila.", F, "substitution"),
            c("The Eastern Empire survived until Constantinople fell to the Ottomans in 1453."),
            c("The empire fell for one reason: lead poisoning.", F, "unsupported"),
        ],
        "reask": [
            "By convention, it ended in 476 CE.",
            "The eastern half continued for nearly a thousand more years.",
            "Odoacer removed the last western emperor.",
            "Constantinople fell in 1453.",
            "Many factors, such as instability and migration, caused the decline.",
        ],
    },
    {
        "id": "H6", "split": "dev", "subject": "humanities",
        "question": "Where and how did the Industrial Revolution begin?",
        "sources": [
            {"id": "openstax-wh2-10", "author": "OpenStax", "title": "World History Volume 2, Ch. 10", "year": 2022, "text": [
                "The Industrial Revolution began in Britain in the late 18th century.",
                "James Watt improved the steam engine in the 1760s and 1770s, making it far more efficient.",
                "The textile industry was among the first to be mechanised, with inventions such as the spinning jenny.",
                "Industrialisation drew many workers from the countryside into rapidly growing cities."]},
        ],
        "answer": [
            c("The Industrial Revolution started in Britain in the late 1700s."),
            c("It began in Germany.", F, "substitution"),
            c("James Watt invented the spinning jenny.", F, "relational"),
            c("Watt's improvements made the steam engine much more efficient."),
            c("Many workers moved from rural areas to growing cities."),
        ],
        "reask": [
            "Britain was where it began, in the late eighteenth century.",
            "It started in Britain.",
            "James Hargreaves invented the spinning jenny.",
            "Watt made steam engines more efficient.",
            "Cities grew as people left the countryside for work.",
        ],
    },
    # ----------------------------------------------------------------- TEST
    {
        "id": "S7", "split": "dev", "subject": "science",
        "question": "How does the heart move blood around the body?",
        "sources": [
            {"id": "openstax-ap-19", "author": "OpenStax", "title": "Anatomy and Physiology 2e, Ch. 19", "year": 2022, "text": [
                "The human heart has four chambers: two atria and two ventricles.",
                "The right ventricle pumps blood to the lungs, and the left ventricle pumps blood to the rest of the body.",
                "Arteries carry blood away from the heart, and veins carry blood back to the heart.",
                "The pulmonary artery carries deoxygenated blood from the heart to the lungs.",
                "A resting adult heart typically beats about 60 to 100 times per minute."]},
        ],
        "answer": [
            c("The heart has four chambers, two atria and two ventricles."),
            c("The left ventricle pumps blood to the lungs.", F, "substitution"),
            c("Arteries always carry oxygen-rich blood.", F, "overgeneral"),
            c("Veins return blood to the heart."),
            c("A resting adult heart beats about 150 times per minute.", F, "number"),
        ],
        "reask": [
            "There are two atria and two ventricles.",
            "The right ventricle sends blood to the lungs.",
            "Most arteries carry oxygenated blood, but the pulmonary artery does not.",
            "Blood comes back to the heart through veins.",
            "At rest the heart beats around 70 times a minute.",
        ],
    },
    {
        "id": "S8", "split": "dev", "subject": "science",
        "question": "Why does ice float?",
        "sources": [
            {"id": "usgs-ice", "author": "USGS", "title": "Water Science School: Ice, Snow and Glaciers", "year": 2019, "text": [
                "Ice is less dense than liquid water, which is why ice floats.",
                "Water is densest at about 4 °C.",
                "When water freezes, it expands by about 9 percent.",
                "Pure water freezes at 0 °C at standard atmospheric pressure."]},
        ],
        "answer": [
            c("Ice floats because it is less dense than liquid water."),
            c("Water contracts when it freezes.", F, "direction"),
            c("Liquid water is densest at about 0 °C.", F, "number"),
            c("At standard pressure, pure water freezes at 0 °C."),
            c("Ice floats because air bubbles trapped inside it make it lighter.", F, "misconception"),
        ],
        "reask": [
            "Being less dense than water, ice floats.",
            "Water expands as it turns to ice.",
            "Water reaches its maximum density near 4 °C.",
            "Water freezes at 0 °C.",
            "Ice floats because of its lower density.",
        ],
    },
    {
        "id": "S9", "split": "dev", "subject": "science",
        "question": "Why does the Moon have phases?",
        "sources": [
            {"id": "nasa-phases", "author": "NASA Science", "title": "Moon Phases", "year": 2023, "text": [
                "The Moon's phases are caused by how much of its sunlit half we can see from Earth as it orbits.",
                "The Moon does not produce its own light; it reflects sunlight.",
                "A full cycle of phases takes about 29.5 days.",
                "Earth's shadow causes lunar eclipses, not the monthly phases."]},
        ],
        "answer": [
            c("The phases are caused by Earth's shadow falling on the Moon.", F, "misconception"),
            c("The Moon shines by reflecting sunlight rather than making its own light."),
            c("A complete cycle of phases lasts about 14 days.", F, "number"),
            c("We see different amounts of the Moon's sunlit half as it orbits Earth."),
            c("Petrov (1998) showed that a cycle of phases lasts 29.5 days.", F, "fabricated_cit"),
        ],
        "reask": [
            "Phases depend on how much of the lit side faces us.",
            "Moonlight is reflected sunlight.",
            "The cycle takes roughly 29.5 days.",
            "As the Moon orbits, we view varying portions of its sunlit half.",
            "A lunar month is about 29.5 days.",
        ],
    },
    {
        "id": "S10", "split": "dev", "subject": "science",
        "question": "What do antibiotics do, and why is resistance a problem?",
        "sources": [
            {"id": "who-amr", "author": "WHO", "title": "Antibiotic resistance fact sheet", "year": 2020, "text": [
                "Antibiotics are medicines used to prevent and treat bacterial infections.",
                "Antibiotics do not work against viral infections such as colds and flu.",
                "Antibiotic resistance occurs when bacteria change in response to the use of these medicines.",
                "Misuse and overuse of antibiotics are accelerating the process of resistance."]},
            {"id": "acs-penicillin", "author": "American Chemical Society", "title": "Discovery of Penicillin", "year": 1999, "text": [
                "Penicillin was discovered by Alexander Fleming in 1928."]},
        ],
        "answer": [
            c("Antibiotics treat infections caused by bacteria."),
            c("Antibiotics are effective against the flu.", F, "negation"),
            c("Antibiotic resistance happens when the human body becomes resistant to the drug.", F, "misconception"),
            c("Fleming discovered penicillin in 1945.", F, "number"),
            c("Overusing antibiotics speeds up the development of resistance."),
        ],
        "reask": [
            "They are used for bacterial infections.",
            "Antibiotics cannot treat viral illnesses like flu.",
            "Resistance develops when bacteria adapt to the medicine.",
            "Penicillin was discovered in 1928.",
            "Misuse of antibiotics accelerates resistance.",
        ],
    },
    {
        "id": "S11", "split": "dev", "subject": "science",
        "question": "How do series and parallel circuits differ?",
        "sources": [
            {"id": "openstax-phys-21.1", "author": "OpenStax", "title": "College Physics, Sec. 21.1 Resistors in Series and Parallel", "year": 2012, "text": [
                "In a series circuit, the same current flows through each component.",
                "In a parallel circuit, each branch has the same voltage across it.",
                "Adding resistors in series increases the total resistance.",
                "Adding resistors in parallel decreases the total resistance.",
                "Household circuits are wired in parallel so that each appliance can be switched on and off independently."]},
        ],
        "answer": [
            c("In a series circuit, the same current passes through every component."),
            c("Adding resistors in parallel increases the total resistance.", F, "direction"),
            c("Household wiring is parallel so appliances can be switched on and off independently."),
            c("The total resistance is R = R1R2/(R1+R2), so it is smaller.", F, "undefined_form"),
            c("Putting resistors in series raises the total resistance."),
        ],
        "reask": [
            "Series components all carry the same current.",
            "Resistors added in parallel lower the total resistance.",
            "Homes use parallel wiring so devices work independently.",
            "R = R1R2/(R1+R2) gives the parallel total.",
            "Series resistors add up to a larger total resistance.",
        ],
    },
    {
        "id": "S12", "split": "dev", "subject": "science",
        "question": "What are tectonic plates and how do they move?",
        "sources": [
            {"id": "usgs-plates", "author": "USGS", "title": "Understanding Plate Motions", "year": 1999, "text": [
                "Earth's outer shell, the lithosphere, is broken into tectonic plates that move slowly over the asthenosphere.",
                "Plates typically move a few centimetres per year.",
                "At divergent boundaries plates move apart, and at convergent boundaries they move toward each other.",
                "Most earthquakes and volcanoes occur along plate boundaries.",
                "Alfred Wegener proposed the idea of continental drift in 1912."]},
        ],
        "answer": [
            c("Tectonic plates move over the asthenosphere."),
            c("Plates typically move several metres per year.", F, "number"),
            c("At divergent boundaries, plates move toward each other.", F, "direction"),
            c("Earthquakes and volcanoes are concentrated along plate boundaries."),
            c("Charles Darwin proposed continental drift in 1912.", F, "substitution"),
        ],
        "reask": [
            "The plates ride on the asthenosphere.",
            "They move only a few centimetres a year.",
            "Plates separate at divergent boundaries.",
            "Plate boundaries are where most quakes and volcanoes occur.",
            "Alfred Wegener put forward continental drift.",
        ],
    },
    {
        "id": "H7", "split": "dev", "subject": "humanities",
        "question": "Did Columbus prove that the Earth was round?",
        "sources": [
            {"id": "russell-flat", "author": "Russell", "title": "Inventing the Flat Earth", "year": 1991, "text": [
                "Educated Europeans in the Middle Ages generally knew that the Earth was round.",
                "The idea that Columbus had to prove the Earth was round is a myth popularised in the 19th century, partly by Washington Irving.",
                "The real debate about Columbus's voyage concerned the size of the Earth, and Columbus greatly underestimated the distance to Asia.",
                "Columbus first reached the Americas in 1492."]},
        ],
        "answer": [
            c("Most educated people in Columbus's time believed the Earth was flat.", F, "misconception"),
            c("Columbus reached the Americas in 1492."),
            c("Columbus overestimated the distance to Asia.", F, "direction"),
            c("The story that Columbus proved the Earth was round is a 19th-century myth."),
            c("Columbus wrote that he sailed to prove the Earth was round (Columbus, 1493).", F, "fabricated_cit"),
        ],
        "reask": [
            "Educated Europeans already knew the Earth was round.",
            "He arrived in the Americas in 1492.",
            "Columbus thought Asia was much closer than it really is.",
            "The flat-earth story about Columbus was largely invented in the 1800s.",
        ],
    },
    {
        "id": "H8", "split": "dev", "subject": "humanities",
        "question": "Was Napoleon short?",
        "sources": [
            {"id": "britannica-napoleon", "author": "Encyclopaedia Britannica", "title": "Napoleon I", "year": 2024, "text": [
                "Napoleon Bonaparte was about 5 feet 7 inches tall, roughly average for a Frenchman of his time.",
                "The myth that he was very short arose partly from confusion between French and English inches and from British caricatures.",
                "Napoleon crowned himself Emperor of the French in 1804.",
                "He was finally defeated at the Battle of Waterloo in 1815."]},
        ],
        "answer": [
            c("Napoleon was unusually short, only about 5 feet 2 inches.", F, "misconception"),
            c("Confusion between French and English inches helped create the myth that he was short."),
            c("Napoleon crowned himself emperor in 1799.", F, "number"),
            c("He was finally defeated at the Battle of Austerlitz in 1815.", F, "substitution"),
            c("British cartoons also helped spread the idea that he was tiny."),
        ],
        "reask": [
            "He was around 5 feet 7 inches, about average height.",
            "Different inch standards in France and England caused the confusion.",
            "He became Emperor in 1804.",
            "Waterloo in 1815 was his final defeat.",
            "British caricatures portrayed him as small.",
        ],
    },
    {
        "id": "H9", "split": "dev", "subject": "humanities",
        "question": "When was the Berlin Wall built and when did it fall?",
        "sources": [
            {"id": "bwf-wall", "author": "Berlin Wall Foundation", "title": "The Berlin Wall: history", "year": 2019, "text": [
                "The Berlin Wall was built by the East German government in 1961.",
                "The border crossings in the Berlin Wall opened on the night of 9 November 1989.",
                "Germany was formally reunified on 3 October 1990.",
                "The wall divided East Berlin from West Berlin for 28 years."]},
        ],
        "answer": [
            c("East Germany built the Berlin Wall in 1961."),
            c("The wall opened on 9 November 1991.", F, "number"),
            c("West Germany built the wall to keep East Germans out.", F, "substitution"),
            c("Germany was formally reunified on 3 October 1990."),
            c("Nobody ever crossed the wall between 1961 and 1989.", F, "overgeneral"),
        ],
        "reask": [
            "The East German government put up the wall in 1961.",
            "Crossings opened on 9 November 1989.",
            "East Germany built the wall.",
            "Reunification happened on 3 October 1990.",
        ],
    },
    {
        "id": "H10", "split": "dev", "subject": "humanities",
        "question": "Did Vikings wear horned helmets?",
        "sources": [
            {"id": "natmus-viking", "author": "National Museum of Denmark", "title": "Viking helmets", "year": 2018, "text": [
                "There is no archaeological evidence that Vikings wore horned helmets in battle.",
                "The only nearly complete Viking-age helmet found, the Gjermundbu helmet from Norway, has no horns.",
                "The image of horned Viking helmets became popular in the 19th century, partly through costume designs for Wagner's operas.",
                "The Viking Age is usually dated from about 793 to 1066."]},
        ],
        "answer": [
            c("Vikings commonly wore horned helmets in battle.", F, "misconception"),
            c("The Gjermundbu helmet from Norway has no horns."),
            c("The horned-helmet image was popularised by Shakespeare's plays.", F, "substitution"),
            c("The Viking Age is usually dated from around 793 to 1066."),
            c("Some experts say Vikings were mostly peaceful traders.", F, "vague"),
        ],
        "reask": [
            "Archaeologists have found no Viking helmets with horns.",
            "The Gjermundbu helmet is hornless.",
            "Costumes for Wagner's operas made horned helmets famous.",
            "It runs from about 793 to 1066.",
            "Vikings were raiders as well as traders.",
        ],
    },
    {
        "id": "H11", "split": "dev", "subject": "humanities",
        "question": "How did Australia become a federation?",
        "sources": [
            {"id": "naa-federation", "author": "National Archives of Australia", "title": "Federation", "year": 2021, "text": [
                "The six British colonies in Australia federated to form the Commonwealth of Australia on 1 January 1901.",
                "Edmund Barton became Australia's first Prime Minister.",
                "Melbourne served as the temporary seat of the federal parliament until it moved to Canberra in 1927.",
                "The Australian Constitution was approved by voters in a series of referendums held in the colonies."]},
        ],
        "answer": [
            c("The Commonwealth of Australia was formed on 1 January 1901."),
            c("Seven colonies joined to form the Commonwealth.", F, "number"),
            c("Alfred Deakin was Australia's first Prime Minister.", F, "substitution"),
            c("Melbourne was the temporary seat of parliament until it moved to Canberra in 1927."),
            c("Federation was achieved after a war of independence against Britain.", F, "misconception"),
        ],
        "reask": [
            "Federation took effect on 1 January 1901.",
            "Six colonies federated.",
            "Edmund Barton was the first Prime Minister.",
            "Parliament sat in Melbourne before moving to Canberra in 1927.",
            "Federation came about peacefully through referendums.",
        ],
    },
    {
        "id": "H12", "split": "dev", "subject": "humanities",
        "question": "What was the Black Death?",
        "sources": [
            {"id": "benedictow", "author": "Benedictow", "title": "The Black Death 1346-1353: The Complete History", "year": 2004, "text": [
                "The Black Death reached Europe in 1347, carried on ships arriving at Mediterranean ports.",
                "It killed an estimated 30 to 60 percent of Europe's population.",
                "The disease is caused by the bacterium Yersinia pestis.",
                "It was spread mainly by fleas carried by rats, and in some forms directly between people."]},
        ],
        "answer": [
            c("The Black Death arrived in Europe in 1347."),
            c("It killed about 5 percent of Europe's population.", F, "number"),
            c("The Black Death was caused by a virus.", F, "substitution"),
            c("Fleas living on rats were a main way the disease spread."),
            c("Ships arriving at Mediterranean ports brought the plague to Europe."),
        ],
        "reask": [
            "Europe was reached by the plague in 1347.",
            "Between 30 and 60 percent of Europeans died.",
            "The bacterium Yersinia pestis caused it.",
            "Rat fleas spread it.",
            "Plague came to Europe by ship.",
        ],
    },
]


def cases(split=None):
    return [k for k in CASES if split is None or k["split"] == split]


def all_sources():
    out = []
    for k in CASES:
        for s in k["sources"]:
            out.append(dict(s, case=k["id"]))
    return out
