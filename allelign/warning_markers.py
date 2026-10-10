"""Stable fragments of the warnings and log reasons allelign emits.

Consumers that classify messages (SMARTpred's verdict, for one) import these
constants instead of copying the text, so a rewording cannot silently stop a
classifier from firing. The values match ldpred3's markers of the same name.
"""

AF_ANTI_CORRELATED = "anti-correlated"
AF_SAFETY_LIMIT = "safety limit"
AF_CONCORDANCE_GUARD = "concordance guard"
LD_SCREEN_HIGH_DROP = "LD-consistency screen dropped"
LD_SCREEN_UNTESTED = "never entered a window"

ATTENTION_MARKERS = (AF_ANTI_CORRELATED, AF_SAFETY_LIMIT, AF_CONCORDANCE_GUARD,
                     LD_SCREEN_HIGH_DROP, LD_SCREEN_UNTESTED)
