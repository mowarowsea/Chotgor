from backend.services.scenario_chat.service import parse_intro_to_turns


def test_parse_intro_header_only_npc_block():
    blocks = parse_intro_to_turns(
        intro_text="@Reika:\nhello\nsecond line\n",
        user_alias="Player",
        known_npc_names={"Reika": "id-r"},
    )

    assert blocks == [
        {
            "speaker_type": "npc",
            "speaker_id": "id-r",
            "speaker_name": "Reika",
            "content": "hello\nsecond line",
        }
    ]


def test_parse_intro_header_only_narrator_block():
    blocks = parse_intro_to_turns(
        intro_text="@Narrator:\nrain falls\nquietly\n",
        user_alias="Player",
        known_npc_names={},
    )

    assert blocks == [
        {
            "speaker_type": "narrator",
            "speaker_id": None,
            "speaker_name": "Narrator",
            "content": "rain falls\nquietly",
        }
    ]
