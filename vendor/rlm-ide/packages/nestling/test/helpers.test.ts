import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { ageNote, ageRangeOf, fileFitsAge, openLong, rankFiles, splitSections } from "../src/library.ts";
import { quantitiesIn, repeatsEarlier, stripWorkMarkup, unsupportedNumbers } from "../src/speak.ts";

function library(files: Record<string, string>): string {
  const dir = mkdtempSync(path.join(tmpdir(), "nestling-lib-"));
  for (const [name, text] of Object.entries(files)) writeFileSync(path.join(dir, name), text);
  return dir;
}

describe("age ranges in headings", () => {
  it("reads a range and runs it to the end of its last unit", () => {
    expect(ageRangeOf("Feeding 7–9 months — expanding variety")).toEqual({ lo: 7, hi: 10 });
    expect(ageRangeOf("Growth 1 to 3 years")).toEqual({ lo: 12, hi: 48 });
    expect(ageRangeOf("Starting solids at 6 months")).toBeNull();
  });

  it("names the sections written for the child's age", () => {
    const sections = splitSections("# T\n## Sleep 0–3 months\na\n## Sleep 4–7 months\nb\n## Sleep 8–12 months\nc\n");
    expect(ageNote(sections, 7.1).note).toContain('"Sleep 4–7 months"');
    expect(ageNote(sections, undefined).note).toBe("");
  });

  it("sets aside a file whose every age range excludes the child", () => {
    const root = library({
      "toddler.md": "# Feeding for 2 to 6 year olds\n## Feeding 2–4 years\nx\n## Feeding 4–6 years\ny\n",
      "infant.md": "# Infant feeding\n## Feeding 0–3 months\nx\n## Feeding 7–9 months\ny\n",
      "general.md": "# Safety\n## Water\nx\n",
    });
    expect(fileFitsAge(path.join(root, "toddler.md"), 7.1)).toBe(false);
    expect(fileFitsAge(path.join(root, "infant.md"), 7.1)).toBe(true);
    expect(fileFitsAge(path.join(root, "general.md"), 7.1)).toBeNull();
    expect(fileFitsAge(path.join(root, "toddler.md"), undefined)).toBeNull();
  });
});

describe("library search", () => {
  it("ranks the file whose text answers the question first, not the longest one", () => {
    const filler = "General advice about babies and parents. ".repeat(400);
    const root = library({
      "long.md": `# Long book\n## Chapter\n${filler}\n## Another\n${filler} milk\n`,
      "express.md": "# Leaflet\n## Express breast milk by hand\nWash hands. Press the areola gently and express the milk into a clean cup.\n",
    });
    const ranked = rankFiles(root, ["long.md", "express.md"], "how do I express breast milk by hand");
    expect(ranked[0]!.file).toBe("express.md");
  });

  it("returns the sections that match from a long file, with the age note first", () => {
    const text = `# Book\n## Feeding 0–3 months\nmilk only\n## Feeding 7–9 months\nthree meals of mashed food\n## Sleep\n${"z ".repeat(3000)}`;
    const out = openLong("book.md", text, "meals", 2000, 7.1);
    expect(out).toContain("Feeding 7–9 months");
    expect(out).toContain("three meals of mashed food");
    expect(out.indexOf("The child is 7.1 months old")).toBeLessThan(out.indexOf("three meals"));
  });
});

describe("what the parent is told", () => {
  it("keeps the words and drops the markup", () => {
    expect(stripWorkMarkup("# Title\n**Iron-rich foods:** meat\n* one")).toBe("Title\nIron-rich foods: meat\n• one");
  });

  it("finds numbers with what they measure, including ranges and spelled-out counts", () => {
    expect(quantitiesIn("4–6 months")).toEqual([{ n: 4, unit: "month" }, { n: 6, unit: "month" }]);
    expect(unsupportedNumbers("Offer three meals and two snacks.", "2–3 meals per day")).toEqual(["2 snack"]);
  });

  it("does not let a number backed as one thing pass as another", () => {
    expect(unsupportedNumbers("She needs 7 kg of food", "She is 7 months old")).toEqual(["7 kg"]);
    expect(unsupportedNumbers("She is at the 40th centile", '{"centile": 39.8}')).toEqual([]);
  });

  it("treats the same answer to a different question as a repeat, but not to the same question", () => {
    const answer = "Darya should eat two to three meals a day with vegetables fruits grains and protein foods and iron rich foods daily.";
    const before = [{ question: "What should she eat?", answer }];
    expect(repeatsEarlier(answer, "How should she sleep at night?", before)).toBe(true);
    expect(repeatsEarlier(answer, "What should she eat?", before)).toBe(false);
  });
});
