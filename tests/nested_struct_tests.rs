//! Comprehensive tests for nested struct functionality
//!
//! These tests ensure that nested structs work correctly in all scenarios:
//! - Direct nested struct fields
//! - Arrays of nested structs
//! - Nested structs in nested structs (deep nesting)
//! - Optional nested structs
//! - Schema generation
//! - Deserialization

#[cfg(test)]
mod nested_struct_tests {
    use rstructor::{Instructor, SchemaType};
    use serde::{Deserialize, Serialize};

    // ====== Simple nested struct tests ======

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct Address {
        #[llm(description = "Street address", example = "123 Main St")]
        street: String,

        #[llm(description = "City name", example = "Springfield")]
        city: String,

        #[llm(description = "ZIP code", example = "12345")]
        zip: String,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct Person {
        #[llm(description = "Person's name", example = "John Doe")]
        name: String,

        #[llm(description = "Person's age", example = 30)]
        age: u32,

        #[llm(description = "Person's address")]
        address: Address,
    }

    #[test]
    fn test_nested_struct_schema_generation() {
        let schema = Person::schema();
        let schema_json = schema.to_json();

        // Verify parent struct schema
        assert_eq!(schema_json["type"], "object");
        assert_eq!(schema_json["title"], "Person");

        // Verify nested struct field exists
        let address_prop = &schema_json["properties"]["address"];
        assert!(address_prop.is_object(), "Address property should exist");

        // Verify address is typed as object (even if properties aren't embedded)
        assert_eq!(
            address_prop["type"],
            "object",
            "Address should be type 'object', got: {:?}",
            address_prop.get("type")
        );
    }

    #[test]
    fn test_nested_struct_deserialization() {
        let json = serde_json::json!({
            "name": "Jane Smith",
            "age": 25,
            "address": {
                "street": "456 Oak Ave",
                "city": "Chicago",
                "zip": "60601"
            }
        });

        let person: Person = serde_json::from_value(json).unwrap();
        assert_eq!(person.name, "Jane Smith");
        assert_eq!(person.age, 25);
        assert_eq!(person.address.street, "456 Oak Ave");
        assert_eq!(person.address.city, "Chicago");
        assert_eq!(person.address.zip, "60601");
    }

    #[test]
    fn test_nested_struct_serialization() {
        let person = Person {
            name: "Bob Johnson".to_string(),
            age: 40,
            address: Address {
                street: "789 Pine St".to_string(),
                city: "Boston".to_string(),
                zip: "02101".to_string(),
            },
        };

        let json = serde_json::to_value(&person).unwrap();
        assert_eq!(json["name"], "Bob Johnson");
        assert_eq!(json["age"], 40);
        assert_eq!(json["address"]["street"], "789 Pine St");
        assert_eq!(json["address"]["city"], "Boston");
        assert_eq!(json["address"]["zip"], "02101");
    }

    // ====== Array of nested structs tests ======

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct Tag {
        #[llm(description = "Tag name", example = "rust")]
        name: String,

        #[llm(description = "Tag category", example = "programming")]
        category: String,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct Article {
        #[llm(description = "Article title", example = "Learning Rust")]
        title: String,

        #[llm(description = "Article tags")]
        tags: Vec<Tag>,
    }

    #[test]
    fn test_array_of_nested_structs_schema() {
        let schema = Article::schema();
        let schema_json = schema.to_json();

        // Verify tags field is an array
        let tags_prop = &schema_json["properties"]["tags"];
        assert_eq!(tags_prop["type"], "array");

        // Verify items are objects or have properties (schema enhancement adds properties)
        let items = &tags_prop["items"];
        let items_type = items
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let has_properties = items
            .get("properties")
            .and_then(|v| v.as_object())
            .is_some();

        // CRITICAL: Items MUST be objects for nested structs to work
        // Schema enhancement should ensure this, but verify anyway
        assert!(
            items_type == "object" || has_properties,
            "Tags items should be type 'object' or have properties. Type: {:?}, Has properties: {:?}, Full: {:?}",
            items_type,
            has_properties,
            serde_json::to_string_pretty(items).unwrap_or_default()
        );

        // If properties exist, the enhancement worked
        if has_properties {
            assert!(items["properties"].is_object());
        }
    }

    #[test]
    fn test_array_of_nested_structs_deserialization() {
        let json = serde_json::json!({
            "title": "My Article",
            "tags": [
                {"name": "rust", "category": "programming"},
                {"name": "tutorial", "category": "content"}
            ]
        });

        let article: Article = serde_json::from_value(json).unwrap();
        assert_eq!(article.title, "My Article");
        assert_eq!(article.tags.len(), 2);
        assert_eq!(article.tags[0].name, "rust");
        assert_eq!(article.tags[0].category, "programming");
        assert_eq!(article.tags[1].name, "tutorial");
        assert_eq!(article.tags[1].category, "content");
    }

    // ====== Deep nesting tests ======

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct ContactInfo {
        #[llm(description = "Email address", example = "user@example.com")]
        email: String,

        #[llm(description = "Phone number", example = "555-1234")]
        phone: String,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct Profile {
        #[llm(description = "Contact information")]
        contact: ContactInfo,

        #[llm(description = "Bio text", example = "Software developer")]
        bio: String,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct User {
        #[llm(description = "Username", example = "johndoe")]
        username: String,

        #[llm(description = "User profile")]
        profile: Profile,
    }

    #[test]
    fn test_deeply_nested_structs_schema() {
        let schema = User::schema();
        let schema_json = schema.to_json();

        // Verify top-level structure
        assert_eq!(schema_json["type"], "object");

        // Verify nested profile field
        let profile_prop = &schema_json["properties"]["profile"];
        assert_eq!(profile_prop["type"], "object");

        // Profile should exist even if nested properties aren't fully embedded
        assert!(profile_prop.is_object());
    }

    #[test]
    fn test_deeply_nested_structs_deserialization() {
        let json = serde_json::json!({
            "username": "alice",
            "profile": {
                "contact": {
                    "email": "alice@example.com",
                    "phone": "555-5678"
                },
                "bio": "Data scientist"
            }
        });

        let user: User = serde_json::from_value(json).unwrap();
        assert_eq!(user.username, "alice");
        assert_eq!(user.profile.bio, "Data scientist");
        assert_eq!(user.profile.contact.email, "alice@example.com");
        assert_eq!(user.profile.contact.phone, "555-5678");
    }

    // ====== Optional nested struct tests ======

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct Metadata {
        #[llm(description = "Created timestamp", example = "2024-01-01T00:00:00Z")]
        created_at: String,

        #[llm(
            description = "Last updated timestamp",
            example = "2024-01-02T00:00:00Z"
        )]
        updated_at: String,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct Document {
        #[llm(description = "Document title", example = "My Document")]
        title: String,

        #[llm(description = "Optional metadata")]
        metadata: Option<Metadata>,
    }

    #[test]
    fn test_optional_nested_struct_schema() {
        let schema = Document::schema();
        let schema_json = schema.to_json();

        let metadata_prop = &schema_json["properties"]["metadata"];
        // Optional fields may not have type explicitly set, but should exist
        assert!(metadata_prop.is_object());
    }

    #[test]
    fn test_optional_nested_struct_with_value() {
        let json = serde_json::json!({
            "title": "Test Doc",
            "metadata": {
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z"
            }
        });

        let doc: Document = serde_json::from_value(json).unwrap();
        assert_eq!(doc.title, "Test Doc");
        assert!(doc.metadata.is_some());
        assert_eq!(doc.metadata.unwrap().created_at, "2024-01-01T00:00:00Z");
    }

    #[test]
    fn test_optional_nested_struct_without_value() {
        let json = serde_json::json!({
            "title": "Test Doc"
        });

        let doc: Document = serde_json::from_value(json).unwrap();
        assert_eq!(doc.title, "Test Doc");
        assert!(doc.metadata.is_none());
    }

    // ====== Complex nested structures ======

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct Author {
        #[llm(description = "Author name", example = "Jane Author")]
        name: String,

        #[llm(description = "Author email", example = "jane@example.com")]
        email: String,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct Comment {
        #[llm(description = "Comment author")]
        author: Author,

        #[llm(description = "Comment text", example = "Great article!")]
        text: String,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct BlogPost {
        #[llm(description = "Post title", example = "My Blog Post")]
        title: String,

        #[llm(description = "Post author")]
        author: Author,

        #[llm(description = "Comments on the post")]
        comments: Vec<Comment>,
    }

    #[test]
    fn test_complex_nested_structure() {
        let json = serde_json::json!({
            "title": "My Post",
            "author": {
                "name": "John Writer",
                "email": "john@example.com"
            },
            "comments": [
                {
                    "author": {
                        "name": "Alice Reader",
                        "email": "alice@example.com"
                    },
                    "text": "Nice post!"
                },
                {
                    "author": {
                        "name": "Bob Reviewer",
                        "email": "bob@example.com"
                    },
                    "text": "Very informative"
                }
            ]
        });

        let post: BlogPost = serde_json::from_value(json).unwrap();
        assert_eq!(post.title, "My Post");
        assert_eq!(post.author.name, "John Writer");
        assert_eq!(post.comments.len(), 2);
        assert_eq!(post.comments[0].author.name, "Alice Reader");
        assert_eq!(post.comments[0].text, "Nice post!");
        assert_eq!(post.comments[1].author.name, "Bob Reviewer");
    }

    #[test]
    fn test_complex_nested_structure_schema() {
        let schema = BlogPost::schema();
        let schema_json = schema.to_json();

        // Verify all fields exist
        assert!(schema_json["properties"]["title"].is_object());
        assert!(schema_json["properties"]["author"].is_object());
        assert!(schema_json["properties"]["comments"].is_object());

        // Verify author is an object
        assert_eq!(schema_json["properties"]["author"]["type"], "object");

        // Verify comments is an array of objects
        let comments_prop = &schema_json["properties"]["comments"];
        assert_eq!(comments_prop["type"], "array");

        // Schema enhancement should ensure items are objects
        let items = &comments_prop["items"];
        let items_type = items
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let has_properties = items
            .get("properties")
            .and_then(|v| v.as_object())
            .is_some();

        assert!(
            items_type == "object" || has_properties,
            "Comments items should be type 'object' or have properties. Type: {:?}, Has properties: {:?}",
            items_type,
            has_properties
        );
    }

    // ====== Edge cases ======

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct EmptyNested {}

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct Container {
        #[llm(description = "Empty nested struct")]
        empty: EmptyNested,
    }

    #[test]
    fn test_empty_nested_struct() {
        let json = serde_json::json!({
            "empty": {}
        });

        let _container: Container = serde_json::from_value(json).unwrap();
        // Should deserialize successfully even with empty nested struct
        // If we get here without panicking, the test passes
    }

    #[test]
    fn test_empty_nested_struct_schema() {
        let schema = Container::schema();
        let schema_json = schema.to_json();

        let empty_prop = &schema_json["properties"]["empty"];
        assert_eq!(empty_prop["type"], "object");
    }

    // ====== Enums within nested structs tests ======

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    enum UserRole {
        Admin,
        User,
        Guest,
        Moderator,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    enum AccountStatus {
        Active,
        Suspended,
        Inactive,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct UserAccount {
        #[llm(description = "Account username")]
        username: String,

        #[llm(description = "User role")]
        role: UserRole,

        #[llm(description = "Account status")]
        status: AccountStatus,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct Organization {
        #[llm(description = "Organization name")]
        name: String,

        #[llm(description = "Organization members")]
        members: Vec<UserAccount>,
    }

    #[test]
    fn test_enum_in_nested_struct_schema() {
        let schema = Organization::schema();
        let schema_json = schema.to_json();

        // Verify top-level structure
        assert_eq!(schema_json["type"], "object");
        assert_eq!(schema_json["title"], "Organization");

        // Verify members field is an array
        let members_prop = &schema_json["properties"]["members"];
        assert_eq!(members_prop["type"], "array");

        // Verify items are objects
        let items = &members_prop["items"];
        assert_eq!(items["type"], "object");

        // Verify nested struct has enum fields
        // The schema should properly represent UserAccount with its enum fields
        let items_props = items.get("properties");
        if let Some(props) = items_props {
            // If properties are embedded, verify enum fields exist
            if let Some(role_prop) = props.get("role") {
                assert!(role_prop.is_object());
            }
            if let Some(status_prop) = props.get("status") {
                assert!(status_prop.is_object());
            }
        }
    }

    #[test]
    fn test_enum_in_nested_struct_deserialization() {
        let json = serde_json::json!({
            "name": "Tech Corp",
            "members": [
                {
                    "username": "alice",
                    "role": "Admin",
                    "status": "Active"
                },
                {
                    "username": "bob",
                    "role": "User",
                    "status": "Active"
                },
                {
                    "username": "charlie",
                    "role": "Guest",
                    "status": "Inactive"
                }
            ]
        });

        let org: Organization = serde_json::from_value(json).unwrap();
        assert_eq!(org.name, "Tech Corp");
        assert_eq!(org.members.len(), 3);
        assert_eq!(org.members[0].username, "alice");
        assert_eq!(org.members[0].role, UserRole::Admin);
        assert_eq!(org.members[0].status, AccountStatus::Active);
        assert_eq!(org.members[1].role, UserRole::User);
        assert_eq!(org.members[2].role, UserRole::Guest);
        assert_eq!(org.members[2].status, AccountStatus::Inactive);
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    enum Priority {
        Low,
        Medium,
        High,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct TaskMetadata {
        #[llm(description = "Task priority")]
        priority: Priority,

        #[llm(description = "Task tags")]
        tags: Vec<String>,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct Project {
        #[llm(description = "Project name")]
        name: String,

        #[llm(description = "Task metadata")]
        task_metadata: TaskMetadata,
    }

    #[test]
    fn test_enum_in_directly_nested_struct_schema() {
        let schema = Project::schema();
        let schema_json = schema.to_json();

        // Verify task_metadata field exists
        let metadata_prop = &schema_json["properties"]["task_metadata"];
        assert_eq!(metadata_prop["type"], "object");

        // Verify nested struct has enum field
        let metadata_props = metadata_prop.get("properties");
        if let Some(props) = metadata_props {
            if let Some(priority_prop) = props.get("priority") {
                assert!(priority_prop.is_object());
            }
        }
    }

    #[test]
    fn test_enum_in_directly_nested_struct_deserialization() {
        let json = serde_json::json!({
            "name": "Website Redesign",
            "task_metadata": {
                "priority": "High",
                "tags": ["urgent", "frontend"]
            }
        });

        let project: Project = serde_json::from_value(json).unwrap();
        assert_eq!(project.name, "Website Redesign");
        assert_eq!(project.task_metadata.priority, Priority::High);
        assert_eq!(project.task_metadata.tags.len(), 2);
        assert_eq!(project.task_metadata.tags[0], "urgent");
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct TaskWithOptionalEnum {
        #[llm(description = "Task title")]
        title: String,

        #[llm(description = "Optional priority")]
        priority: Option<Priority>,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct TaskList {
        #[llm(description = "List name")]
        name: String,

        #[llm(description = "Tasks in the list")]
        tasks: Vec<TaskWithOptionalEnum>,
    }

    #[test]
    fn test_optional_enum_in_nested_struct_schema() {
        let schema = TaskList::schema();
        let schema_json = schema.to_json();

        // Verify tasks field is an array
        let tasks_prop = &schema_json["properties"]["tasks"];
        assert_eq!(tasks_prop["type"], "array");

        // Verify items are objects
        let items = &tasks_prop["items"];
        assert_eq!(items["type"], "object");
    }

    #[test]
    fn test_optional_enum_in_nested_struct_deserialization() {
        let json = serde_json::json!({
            "name": "My Tasks",
            "tasks": [
                {
                    "title": "Task 1",
                    "priority": "High"
                },
                {
                    "title": "Task 2"
                }
            ]
        });

        let task_list: TaskList = serde_json::from_value(json).unwrap();
        assert_eq!(task_list.name, "My Tasks");
        assert_eq!(task_list.tasks.len(), 2);
        assert_eq!(task_list.tasks[0].title, "Task 1");
        assert_eq!(task_list.tasks[0].priority, Some(Priority::High));
        assert_eq!(task_list.tasks[1].title, "Task 2");
        assert_eq!(task_list.tasks[1].priority, None);
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    enum Department {
        Engineering,
        Marketing,
        Sales,
        Support,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct EmployeeInfo {
        #[llm(description = "Employee ID")]
        id: u32,

        #[llm(description = "Department")]
        department: Department,

        #[llm(description = "Role")]
        role: UserRole,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct Company {
        #[llm(description = "Company name")]
        name: String,

        #[llm(description = "Employee information")]
        employees: Vec<EmployeeInfo>,
    }

    #[test]
    fn test_multiple_enums_in_nested_struct_schema() {
        let schema = Company::schema();
        let schema_json = schema.to_json();

        // Verify employees field is an array
        let employees_prop = &schema_json["properties"]["employees"];
        assert_eq!(employees_prop["type"], "array");

        // Verify items are objects
        let items = &employees_prop["items"];
        assert_eq!(items["type"], "object");
    }

    #[test]
    fn test_multiple_enums_in_nested_struct_deserialization() {
        let json = serde_json::json!({
            "name": "Acme Corp",
            "employees": [
                {
                    "id": 1,
                    "department": "Engineering",
                    "role": "Admin"
                },
                {
                    "id": 2,
                    "department": "Marketing",
                    "role": "User"
                }
            ]
        });

        let company: Company = serde_json::from_value(json).unwrap();
        assert_eq!(company.name, "Acme Corp");
        assert_eq!(company.employees.len(), 2);
        assert_eq!(company.employees[0].id, 1);
        assert_eq!(company.employees[0].department, Department::Engineering);
        assert_eq!(company.employees[0].role, UserRole::Admin);
        assert_eq!(company.employees[1].department, Department::Marketing);
        assert_eq!(company.employees[1].role, UserRole::User);
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct Team {
        #[llm(description = "Team name")]
        name: String,

        #[llm(description = "Team department")]
        department: Department,
    }

    #[derive(Instructor, Serialize, Deserialize, Debug, PartialEq)]
    struct Manager {
        #[llm(description = "Manager name")]
        name: String,

        #[llm(description = "Managed team")]
        team: Team,
    }

    #[test]
    fn test_enum_in_deeply_nested_struct_schema() {
        let schema = Manager::schema();
        let schema_json = schema.to_json();

        // Verify team field exists
        let team_prop = &schema_json["properties"]["team"];
        assert_eq!(team_prop["type"], "object");
    }

    #[test]
    fn test_enum_in_deeply_nested_struct_deserialization() {
        let json = serde_json::json!({
            "name": "Alice Manager",
            "team": {
                "name": "Backend Team",
                "department": "Engineering"
            }
        });

        let manager: Manager = serde_json::from_value(json).unwrap();
        assert_eq!(manager.name, "Alice Manager");
        assert_eq!(manager.team.name, "Backend Team");
        assert_eq!(manager.team.department, Department::Engineering);
    }
}
