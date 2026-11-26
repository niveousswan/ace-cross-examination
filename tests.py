"""Unit checks for the ACE probes. Run: python tests.py"""
import engine as E
assert E.number_conflict("Fleming discovered penicillin in 1945.", "Penicillin was discovered by Alexander Fleming in 1928.")
assert not E.number_conflict("At sea level water boils at 100 °C.", "At sea level, pure water boils at 100 °C.")
assert E.number_conflict("It killed about 5 percent.", "It killed an estimated 30 to 60 percent of Europe's population.")
assert not E.number_conflict("About 40 percent died.", "It killed an estimated 30 to 60 percent of Europe's population.")
assert E.direction_conflict("Water contracts when it freezes.", "When water freezes, it expands by about 9 percent.")
assert E.negation_conflict("Sound can travel through the vacuum of space.", "Sound cannot travel through a vacuum.")
assert E.contrast_conflict("In DNA, adenine pairs with uracil.", "In DNA, adenine pairs with thymine and guanine pairs with cytosine.")
assert E.extract_citations("Sound is fast (Harris, 2003).")[0] == [("Harris", 2003)]
assert len(E.split_claims("It is 9.8 m/s². The Moon has 1.6 m/s². Done.")) == 3
assert E.find_conflict("Putting resistors in series raises the total resistance.",
                       "Adding resistors in parallel decreases the total resistance.") is None  # different statement
print("all probe tests passed")
