#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const rootDir = path.resolve(__dirname, "..");
const notesDir = path.join(rootDir, "notes");
const outputFile = path.join(rootDir, "notes-data.js");
const requiredFields = ["id", "title", "category", "date", "readTime", "tags", "summary"];

function parseFrontmatterValue(rawValue) {
  const trimmedValue = rawValue.trim();

  if (!trimmedValue) {
    return "";
  }

  try {
    return JSON.parse(trimmedValue);
  } catch (error) {
    return trimmedValue;
  }
}

function parseNoteFile(filePath) {
  const fileContent = fs.readFileSync(filePath, "utf8");
  const frontmatterMatch = fileContent.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);

  if (!frontmatterMatch) {
    throw new Error(`${path.relative(rootDir, filePath)} is missing frontmatter`);
  }

  const metadata = {};
  const frontmatter = frontmatterMatch[1];
  const content = frontmatterMatch[2].trim();

  for (const line of frontmatter.split("\n")) {
    if (!line.trim()) {
      continue;
    }

    const separatorIndex = line.indexOf(":");
    if (separatorIndex === -1) {
      throw new Error(`Invalid frontmatter line in ${path.relative(rootDir, filePath)}: ${line}`);
    }

    const key = line.slice(0, separatorIndex).trim();
    const value = line.slice(separatorIndex + 1);
    metadata[key] = parseFrontmatterValue(value);
  }

  for (const requiredField of requiredFields) {
    if (metadata[requiredField] === undefined || metadata[requiredField] === "") {
      throw new Error(`${path.relative(rootDir, filePath)} is missing required field: ${requiredField}`);
    }
  }

  if (!Array.isArray(metadata.tags)) {
    throw new Error(`${path.relative(rootDir, filePath)} field tags must be an array`);
  }

  return {
    id: metadata.id,
    title: metadata.title,
    category: metadata.category,
    date: metadata.date,
    order: Number(metadata.order || 1000),
    readTime: metadata.readTime,
    tags: metadata.tags,
    summary: metadata.summary,
    ...(metadata.hero ? { hero: metadata.hero } : {}),
    content
  };
}

function buildNotesData() {
  const noteFiles = fs.readdirSync(notesDir)
    .filter(fileName => fileName.endsWith(".md"))
    .sort();

  const notes = noteFiles
    .map(fileName => parseNoteFile(path.join(notesDir, fileName)))
    .sort((firstNote, secondNote) => {
      const dateComparison = secondNote.date.localeCompare(firstNote.date);
      if (dateComparison !== 0) {
        return dateComparison;
      }

      return firstNote.order - secondNote.order;
    });

  const duplicatedIds = notes
    .map(note => note.id)
    .filter((id, index, ids) => ids.indexOf(id) !== index);

  if (duplicatedIds.length) {
    throw new Error(`Duplicated note id: ${Array.from(new Set(duplicatedIds)).join(", ")}`);
  }

  const output = `const NOTES = ${JSON.stringify(notes, null, 2)};\n`;
  fs.writeFileSync(outputFile, output);
  console.log(`Built ${notes.length} notes into ${path.relative(rootDir, outputFile)}`);
}

buildNotesData();
