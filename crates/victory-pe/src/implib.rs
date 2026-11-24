//! COFF import libraries (`.lib`).
//!
//! An umbrella import library tells us which DLL provides each API that a
//! console title may link against. That is the map the runtime needs to
//! decide which of our implementations answers which import.

use object::read::archive::ArchiveFile;
use object::read::coff::{ImportFile, ImportName, ImportType};

use crate::{ImportSymbol, PeError};

/// One `dll!export` pair described by an import library.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct LibImport {
    pub dll: String,
    pub export: ImportSymbol,
    pub kind: LibImportKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LibImportKind {
    Code,
    Data,
    Const,
}

/// Result of reading an import library.
#[derive(Debug, Default)]
pub struct ImportLibrary {
    pub imports: Vec<LibImport>,
    /// Members that are not short-format import objects (regular objects,
    /// long-format import descriptors). Reported so nothing is silently lost.
    pub other_members: usize,
}

impl ImportLibrary {
    pub fn parse(data: &[u8]) -> Result<Self, PeError> {
        let archive = ArchiveFile::parse(data)?;
        let mut lib = ImportLibrary::default();
        for member in archive.members() {
            let member = member?;
            let bytes = member.data(data)?;
            match ImportFile::parse(bytes) {
                Ok(import) => lib.imports.push(LibImport {
                    dll: String::from_utf8_lossy(import.dll()).into_owned(),
                    export: match import.import() {
                        ImportName::Name(name) => ImportSymbol::Name {
                            hint: 0,
                            name: String::from_utf8_lossy(name).into_owned(),
                        },
                        ImportName::Ordinal(o) => ImportSymbol::Ordinal(o),
                    },
                    kind: match import.import_type() {
                        ImportType::Code => LibImportKind::Code,
                        ImportType::Data => LibImportKind::Data,
                        ImportType::Const => LibImportKind::Const,
                    },
                }),
                Err(_) => lib.other_members += 1,
            }
        }
        lib.imports.sort();
        lib.imports.dedup();
        Ok(lib)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A short-format import object (`IMPORT_OBJECT_HEADER` followed by the
    /// symbol and DLL names). `name_type` 0 imports by ordinal (`hint`).
    fn import_object(dll: &str, symbol: &str, hint: u16, kind: u16, name_type: u16) -> Vec<u8> {
        let mut o = Vec::new();
        o.extend_from_slice(&0u16.to_le_bytes()); // Sig1: IMAGE_FILE_MACHINE_UNKNOWN
        o.extend_from_slice(&0xFFFFu16.to_le_bytes()); // Sig2
        o.extend_from_slice(&0u16.to_le_bytes()); // Version
        o.extend_from_slice(&0x8664u16.to_le_bytes()); // Machine: AMD64
        o.extend_from_slice(&0u32.to_le_bytes()); // TimeDateStamp
        let size = symbol.len() + 1 + dll.len() + 1;
        o.extend_from_slice(&(size as u32).to_le_bytes()); // SizeOfData
        o.extend_from_slice(&hint.to_le_bytes()); // Ordinal or hint
        o.extend_from_slice(&(kind | (name_type << 2)).to_le_bytes()); // Type, NameType
        o.extend_from_slice(symbol.as_bytes());
        o.push(0);
        o.extend_from_slice(dll.as_bytes());
        o.push(0);
        o
    }

    /// A `!<arch>` archive of `members`, each with a short name.
    fn archive(members: &[Vec<u8>]) -> Vec<u8> {
        let mut a = b"!<arch>\n".to_vec();
        for (i, m) in members.iter().enumerate() {
            let header = format!(
                "{:<16}{:<12}{:<6}{:<6}{:<8}{:<10}`\n",
                format!("m{i}.o/"),
                0,
                0,
                0,
                644,
                m.len()
            );
            assert_eq!(header.len(), 60);
            a.extend_from_slice(header.as_bytes());
            a.extend_from_slice(m);
            if a.len() % 2 == 1 {
                a.push(b'\n');
            }
        }
        a
    }

    fn by_name(dll: &str, name: &str, kind: LibImportKind) -> LibImport {
        LibImport {
            dll: dll.into(),
            export: ImportSymbol::Name {
                hint: 0,
                name: name.into(),
            },
            kind,
        }
    }

    #[test]
    fn reads_code_data_const_and_ordinal_imports() {
        let lib = archive(&[
            import_object("b.dll", "Beta", 3, 0, 1),
            import_object("a.dll", "Alpha", 0, 0, 1),
            import_object("a.dll", "Table", 0, 1, 1),
            import_object("a.dll", "Limit", 0, 2, 1),
            import_object("a.dll", "Ord", 42, 0, 0),
        ]);
        let parsed = ImportLibrary::parse(&lib).unwrap();
        assert_eq!(parsed.other_members, 0);
        assert_eq!(
            parsed.imports,
            vec![
                by_name("a.dll", "Alpha", LibImportKind::Code),
                by_name("a.dll", "Limit", LibImportKind::Const),
                by_name("a.dll", "Table", LibImportKind::Data),
                LibImport {
                    dll: "a.dll".into(),
                    export: ImportSymbol::Ordinal(42),
                    kind: LibImportKind::Code,
                },
                by_name("b.dll", "Beta", LibImportKind::Code),
            ]
        );
    }

    #[test]
    fn duplicates_are_merged_and_other_members_counted() {
        let lib = archive(&[
            import_object("a.dll", "Alpha", 0, 0, 1),
            b"not an import object".to_vec(),
            import_object("a.dll", "Alpha", 0, 0, 1),
        ]);
        let parsed = ImportLibrary::parse(&lib).unwrap();
        assert_eq!(
            parsed.imports,
            vec![by_name("a.dll", "Alpha", LibImportKind::Code)]
        );
        assert_eq!(parsed.other_members, 1);
    }

    #[test]
    fn an_empty_archive_has_no_imports() {
        let parsed = ImportLibrary::parse(b"!<arch>\n").unwrap();
        assert!(parsed.imports.is_empty());
        assert_eq!(parsed.other_members, 0);
    }

    #[test]
    fn rejects_garbage_without_panicking() {
        assert!(ImportLibrary::parse(b"").is_err());
        assert!(ImportLibrary::parse(b"MZ, not an archive").is_err());
        let mut truncated = archive(&[import_object("a.dll", "Alpha", 0, 0, 1)]);
        truncated.truncate(40);
        assert!(ImportLibrary::parse(&truncated).is_err());
    }
}
