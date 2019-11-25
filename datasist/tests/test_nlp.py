from datasist import nlp


# Corpus for testing pre-processing
sent = u"14000000 8888 I've been meaning to write this down anyways (in case you're interested and for everyone else who comes across this issue) Going forward, we're actually thinking about encouraging the import syntax more, or even making it the default, recommended way of loading models."


def test_pre_process():
    expected = '[#####, ####, meaning, write, anyways, (, case, interested, comes, issue, ), Going, forward, ,, actually, thinking, encouraging, import, syntax, ,, making, default, ,, recommended, way, loading, models, .]'
    output = nlp.pre_process(sent, True, False, True)
    assert expected == output